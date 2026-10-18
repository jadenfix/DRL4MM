"""Off-policy actor-critic training of the v2 quoting policy (TD3 or SAC).

Observation: 16 causal market features + 9 agent-state features (sim.py), z-scored with
statistics fitted on training days only. Action u in [-1, 1]^2:
half-spread = half_max x ((u0 + 1) / 2)^2 bps, skew = skew_max x u1 bps, quotes placed around
the microprice (or the mid, for the ablation). Reward per one-second step, in bps of the quote
notional: mark-to-mid PnL change - eta x (inventory in quotes)^2, plus the liquidation cost on
the final step (reward='pnl'); or, with reward='markout', each fill's markout against the mid
``markout_bins`` later instead of the PnL change (rewards may use the future; observations never do). Episodes are clock hours drawn from the BTC/ETH/LTC training days (domain
randomisation across instruments and regimes); SOL is never loaded here.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TRAIN_SYMBOLS, split_days
from .policies import LearnedActor, NumpyMLP, run_policy
from .sim import N_MARKET, N_OBS, Day, Episode, UNIT

DEFAULTS = dict(
    algo='td3', seed=0, steps=600_000, gamma=0.99, eta=0.0, anchor='micro', features='full',
    half_max=30.0, skew_max=10.0, lr=3e-4, batch=256, start=20_000, buffer=600_000, tau=0.005,
    actor_hidden=[64, 64], critic_hidden=[128, 128], noise=0.2, noise_final=0.05,
    policy_delay=2, target_noise=0.2, target_clip=0.5, eval_every=50_000, latency=1, cache_days=6,
    reward='pnl', markout_bins=300, action='quadratic', h0=7.5,
)
AGENT_STATS = ([0.0, 3.0, 0.5, 0.5, 0.0, 2.5, 0.5, 0.0, 2.5],   # mean of the 9 agent features
               [2.5, 2.0, 0.29, 0.5, 3.0, 2.5, 0.5, 3.0, 2.5])  # scale
BASIC = [0, 1, 3, 10]  # 'basic' feature set: spread (ticks, bps), tick size, 60 s volatility


def day_path(data, sym, day):
    return Path(data) / f'{sym}_{day}.npz'


def fit_normalizer(data_dir, symbols=TRAIN_SYMBOLS):
    rows = []
    for sym in symbols:
        for d in split_days('train'):
            day = Day(day_path(data_dir, sym, d))
            f = day.features()
            rows.append(f[1 + day.first_bin // 10::7].astype(np.float64))
    X = np.concatenate(rows)
    mean, std = X.mean(0), X.std(0)
    std[std < 1e-8] = 1.0
    return {'mean': mean.tolist() + AGENT_STATS[0], 'std': std.tolist() + AGENT_STATS[1], 'clip': 5.0,
            'fitted_on': f'{len(X)} decision rows of the training days of {symbols}'}


def feature_mask(kind):
    m = np.ones(N_OBS, np.float32)
    if kind == 'basic':
        m[:N_MARKET] = 0.0
        m[BASIC] = 1.0
    return m


def mlp(i, hidden, o):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, hidden, sac=False):
        super().__init__()
        self.sac = sac
        self.net = mlp(N_OBS, hidden, 4 if sac else 2)

    def forward(self, x):
        out = self.net(x)
        return torch.tanh(out[:, :2])

    def sample(self, x):
        out = self.net(x)
        mu, log_std = out[:, :2], out[:, 2:].clamp(-5, 1)
        std = log_std.exp()
        z = mu + std * torch.randn_like(mu)
        a = torch.tanh(z)
        logp = (-0.5 * ((z - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(1)
        logp -= (2 * (math.log(2) - z - F.softplus(-2 * z))).sum(1)
        return a, logp

    def export(self):
        lin = [m for m in self.net if isinstance(m, nn.Linear)]
        out = []
        for i, L in enumerate(lin):
            w, b = L.weight.detach().cpu().numpy(), L.bias.detach().cpu().numpy()
            if i == len(lin) - 1:
                w, b = w[:2], b[:2]  # SAC: deterministic action = tanh(mean)
            out.append({'in': int(w.shape[1]), 'out': int(w.shape[0]), 'w': w.ravel().tolist(), 'b': b.tolist()})
        return out


class Critic(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.q1, self.q2 = mlp(N_OBS + 2, hidden, 1), mlp(N_OBS + 2, hidden, 1)

    def forward(self, x, a):
        xa = torch.cat([x, a], 1)
        return self.q1(xa), self.q2(xa)


class Buffer:
    def __init__(self, n):
        self.o = np.zeros((n, N_OBS), np.float32)
        self.a = np.zeros((n, 2), np.float32)
        self.r = np.zeros(n, np.float32)
        self.o2 = np.zeros((n, N_OBS), np.float32)
        self.d = np.zeros(n, np.float32)
        self.n, self.i, self.full = n, 0, False

    def add(self, o, a, r, o2, d):
        i = self.i
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.i = (i + 1) % self.n
        self.full |= self.i == 0

    def sample(self, k, rng):
        idx = rng.integers(0, self.n if self.full else self.i, k)
        t = torch.from_numpy
        return t(self.o[idx]), t(self.a[idx]), t(self.r[idx]), t(self.o2[idx]), t(self.d[idx])


class ObsMaker:
    def __init__(self, norm, mask):
        self.mean = np.asarray(norm['mean'], np.float64)
        self.std = np.asarray(norm['std'], np.float64)
        self.clip = norm['clip']
        self.mask = mask

    def __call__(self, ep):
        m = ep.day.features()[ep.t // 10].astype(np.float64)
        x = np.concatenate([m, ep.agent_features()])
        return (np.clip((x - self.mean) / self.std, -self.clip, self.clip) * self.mask).astype(np.float32)


class MaskedMLP(NumpyMLP):
    def __init__(self, layers, mask):
        super().__init__(layers)
        self.mask = mask

    def __call__(self, z):
        return super().__call__(np.asarray(z, np.float32) * self.mask)


def learned_policy(actor_layers, norm, cfg, name='rl_v2'):
    """A LearnedActor from exported layers (numpy forward = what is shipped and tested)."""
    nets = [MaskedMLP(L, feature_mask(cfg['features'])) for L in actor_layers]
    return LearnedActor(nets, norm['mean'], norm['std'], cfg['half_max'], cfg['skew_max'], norm['clip'], name)


def half_spread_bps(u0, cfg):
    """'quadratic': half_max x ((u0 + 1) / 2)^2 (0 .. half_max);
    'residual': h0 x (1 + 0.75 u0) / (1 - 0.75 u0) (h0 / 7 .. 7 h0), so u = 0 is the
    validation-tuned fixed half-spread h0 and the actor learns departures from it.
    Only + - * / are used so the TypeScript port reproduces it bit for bit."""
    if cfg.get('action', 'quadratic') == 'residual':
        return cfg['h0'] * (1.0 + 0.75 * u0) / (1.0 - 0.75 * u0)
    return cfg['half_max'] * ((u0 + 1.0) * 0.5) ** 2


def quotes_for(ep, u, cfg):
    from .sim import quote_prices
    B, BQ, A, AQ = ep.book()
    mid = (B + A) * 0.5
    c = mid if cfg['anchor'] == 'mid' else B + (A - B) * BQ / (BQ + AQ)
    half = half_spread_bps(float(u[0]), cfg)
    bps = 1e-4 * mid
    return quote_prices(mid, c + cfg['skew_max'] * float(u[1]) * bps, half * bps, B, A)


class PolicyCall:
    """Callable policy object for run_policy, honouring the anchor setting."""

    def __init__(self, actor_layers, norm, cfg, name='rl_v2'):
        self.inner = learned_policy(actor_layers, norm, {**cfg, 'anchor': 'micro'}, name)
        self.cfg = cfg
        self.name = name

    def reset(self, ep):
        pass

    def __call__(self, ep, k):
        return quotes_for(ep, self.inner.act(self.inner.obs(ep)), self.cfg)


def evaluate(layers_list, norm, cfg, data_dir, split='val', symbols=TRAIN_SYMBOLS, every=1, name='rl_v2'):
    pol = PolicyCall(layers_list, norm, cfg, name)
    res = []
    for sym in symbols:
        for d in split_days(split):
            day = Day(day_path(data_dir, sym, d))
            for s in day.hours()[::every]:
                res.append(run_policy(day, s, pol, latency_bins=cfg['latency']))
            del day
    return res


def train(cfg, data_dir, out_dir, log=print):
    cfg = {**DEFAULTS, **cfg}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(cfg, indent=1))
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '4')))
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm_path = Path(data_dir) / 'normalizer.json'
    if not norm_path.exists():
        norm_path.write_text(json.dumps(fit_normalizer(data_dir)))
    norm = json.loads(norm_path.read_text())
    (out / 'normalizer.json').write_text(json.dumps(norm))
    mask = feature_mask(cfg['features'])
    obs_of = ObsMaker(norm, mask)
    sac = cfg['algo'] == 'sac'
    actor = Actor(cfg['actor_hidden'], sac)
    if cfg['action'] == 'residual':  # start at u ~ 0, i.e. at the fixed half-spread
        with torch.no_grad():
            last = [m for m in actor.net if isinstance(m, nn.Linear)][-1]
            last.weight.mul_(0.01)
            last.bias.zero_()
    actor_t = Actor(cfg['actor_hidden'], sac)
    actor_t.load_state_dict(actor.state_dict())
    critic, critic_t = Critic(cfg['critic_hidden']), Critic(cfg['critic_hidden'])
    critic_t.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg['lr'])
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg['lr'])
    log_alpha = torch.zeros(1, requires_grad=True)
    opt_al = torch.optim.Adam([log_alpha], lr=cfg['lr'])
    target_ent = -2.0
    buf = Buffer(cfg['buffer'])
    paths = [day_path(data_dir, s, d) for s in TRAIN_SYMBOLS for d in split_days('train')]
    cache = OrderedDict()

    def get_day():
        p = paths[rng.integers(len(paths))]
        if p not in cache:
            if len(cache) >= cfg['cache_days']:
                cache.popitem(last=False)
            cache[p] = Day(p)
        return cache[p]

    gamma, total, n_upd, ep_count = cfg['gamma'], 0, 0, 0
    best, history, t_start = -1e18, [], time.time()
    next_eval = cfg['eval_every']
    while total < cfg['steps']:
        day = get_day()
        hours = day.hours()
        ep = Episode(day, hours[rng.integers(len(hours))], latency_bins=cfg['latency'])
        notional = ep.Q * UNIT * day.mid_ticks[ep.t0] * day.tick
        scale = 1e4 / notional
        o = obs_of(ep)
        ep_ret = 0.0
        while not ep.done:
            if total < cfg['start']:
                if cfg['action'] == 'residual':  # explore around the fixed half-spread
                    a = np.clip(rng.normal(0, 0.5, 2), -1, 1).astype(np.float32)
                else:
                    a = rng.uniform(-1, 1, 2).astype(np.float32)
            else:
                with torch.no_grad():
                    x = torch.from_numpy(o)[None]
                    if sac:
                        a = actor.sample(x)[0][0].numpy()
                    else:
                        frac = min(1.0, total / cfg['steps'])
                        sd = cfg['noise'] + (cfg['noise_final'] - cfg['noise']) * frac
                        a = np.clip(actor(x)[0].numpy() + rng.normal(0, sd, 2), -1, 1).astype(np.float32)
            inv0 = ep.inv / ep.Q
            bid, ask = quotes_for(ep, a, cfg)
            nf = len(ep.fills)
            de = ep.step(bid, ask)
            if ep.done:
                de += ep.liquidation
            if cfg['reward'] == 'markout':
                # each fill is credited with its markout against the mid markout_bins later;
                # inventory risk beyond that is left to the quadratic penalty
                de = (ep.liquidation if ep.done else 0.0)
                for (fb, side, px, units) in ep.fills[nf:]:
                    j = min(fb + cfg['markout_bins'], day.n_bins - 1)
                    de += side * units * UNIT * ((int(day.bid[j]) + int(day.ask[j])) * 0.5 - px) * day.tick
            r = de * scale - cfg['eta'] * inv0 * inv0
            ep_ret += r
            o2 = obs_of(ep) if not ep.done else o
            buf.add(o, a, r, o2, float(ep.done))
            o = o2
            total += 1
            if total >= cfg['start']:
                ob, ab, rb, o2b, db = buf.sample(cfg['batch'], rng)
                with torch.no_grad():
                    if sac:
                        a2, lp2 = actor.sample(o2b)
                        q1t, q2t = critic_t(o2b, a2)
                        qt = torch.min(q1t, q2t).squeeze(1) - log_alpha.exp() * lp2
                    else:
                        noise = (torch.randn_like(ab) * cfg['target_noise']).clamp(-cfg['target_clip'], cfg['target_clip'])
                        a2 = (actor_t(o2b) + noise).clamp(-1, 1)
                        q1t, q2t = critic_t(o2b, a2)
                        qt = torch.min(q1t, q2t).squeeze(1)
                    y = rb + gamma * (1 - db) * qt
                q1, q2 = critic(ob, ab)
                lc = F.mse_loss(q1.squeeze(1), y) + F.mse_loss(q2.squeeze(1), y)
                opt_c.zero_grad()
                lc.backward()
                opt_c.step()
                n_upd += 1
                if sac:
                    an, lp = actor.sample(ob)
                    q1n, q2n = critic(ob, an)
                    la = (log_alpha.exp().detach() * lp - torch.min(q1n, q2n).squeeze(1)).mean()
                    opt_a.zero_grad()
                    la.backward()
                    opt_a.step()
                    lal = -(log_alpha * (lp.detach() + target_ent)).mean()
                    opt_al.zero_grad()
                    lal.backward()
                    opt_al.step()
                    soft_update(critic_t, critic, cfg['tau'])
                elif n_upd % cfg['policy_delay'] == 0:
                    la = -critic(ob, actor(ob))[0].mean()
                    opt_a.zero_grad()
                    la.backward()
                    opt_a.step()
                    soft_update(critic_t, critic, cfg['tau'])
                    soft_update(actor_t, actor, cfg['tau'])
            if total >= next_eval or total >= cfg['steps']:
                break
        ep_count += 1
        if total >= next_eval or total >= cfg['steps']:
            next_eval += cfg['eval_every']
            layers = actor.export()
            res = evaluate([layers], norm, cfg, data_dir, 'val', every=2)
            pnl = np.array([r['pnl'] for r in res])
            per = {s: float(np.mean([r['pnl'] for r in res if r['symbol'] == s])) for s in TRAIN_SYMBOLS}
            rec = {'steps': total, 'episodes': ep_count, 'secs': round(time.time() - t_start),
                   'val_pnl_mean': float(pnl.mean()), 'val_pnl_std': float(pnl.std()), 'val_per_symbol': per,
                   'val_fills': float(np.mean([r['fills'] for r in res])),
                   'val_abs_inv': float(np.mean([r['mean_abs_inv'] for r in res])),
                   'alpha': float(log_alpha.detach().exp()) if sac else None}
            (out / f'actor_{total:07d}.json').write_text(json.dumps(layers))
            if rec['val_pnl_mean'] > best:
                best = rec['val_pnl_mean']
                rec['best'] = True
                (out / 'actor_best.json').write_text(json.dumps(layers))
            history.append(rec)
            (out / 'progress.json').write_text(json.dumps(history, indent=1))
            torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict(), 'steps': total}, out / 'last.pt')
            log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
    return history


def soft_update(tgt, src, tau):
    with torch.no_grad():
        for pt, p in zip(tgt.parameters(), src.parameters()):
            pt.mul_(1 - tau).add_(p, alpha=tau)
