"""Quoting policies: baselines, the v1 TD3 actor through an observation adapter, and the
learned v2 actor. A policy is called once per decision as ``policy(ep, k)`` and returns
(bid_ticks | None, ask_ticks | None). ``reset(ep)`` is called at the start of every episode."""
from __future__ import annotations

import json
import math
from collections import deque

import numpy as np

from .config import DECISION_BINS
from .sim import UNIT, quote_prices


class FixedSpread:
    """Symmetric quotes at mid +/- h bps (h = 0 joins the touch), re-centred every second."""

    def __init__(self, half_bps, anchor='mid'):
        self.h = float(half_bps)
        self.anchor = anchor
        self.name = f'fixed_{anchor}_{self.h:g}bps'

    def reset(self, ep):
        pass

    def __call__(self, ep, k):
        B, BQ, A, AQ = ep.book()
        mid = (B + A) * 0.5
        c = mid if self.anchor == 'mid' else B + (A - B) * BQ / (BQ + AQ)
        return quote_prices(mid, c, self.h * 1e-4 * mid, B, A)


class RepoFixedMidpoint:
    """simulator/replay.py ``fixed-midpoint-quotes``: bid/ask = mid x (1 -/+ b), widened to a
    one-tick minimum spread, rounded to the nearest tick (repo CLI default b = 10 bps)."""

    def __init__(self, bps=10.0):
        self.b = bps * 1e-4
        self.name = f'repo_fixed_midpoint_{bps:g}bps'

    def reset(self, ep):
        pass

    def __call__(self, ep, k):
        B, _, A, _ = ep.book()
        return midpoint_quotes((B + A) * 0.5, self.b, self.b, B, A)


def midpoint_quotes(mid_t, d_bid, d_ask, B, A):
    bid = mid_t - d_bid * mid_t
    ask = mid_t + d_ask * mid_t
    if ask - bid < 1.0:
        adj = (1.0 - (ask - bid)) / 2
        bid -= adj
        ask += adj
    return min(math.floor(bid + 0.5), A - 1), max(math.floor(ask + 0.5), B + 1)


class AvellanedaStoikov:
    """Avellaneda & Stoikov (2008) with inventory skew, in bps of mid and quote units:
    reservation r = mid - q gamma sigma^2 (T - t); total spread gamma sigma^2 (T - t) + (2/gamma) ln(1 + gamma/k).
    sigma (bps / sqrt(s)) and k (1 / bps) are estimated from training days only; gamma is tuned
    on validation days. T - t is the time left in the hourly episode or, for the stationary
    variant (also tuned on validation), a constant horizon of ``horizon`` seconds."""

    def __init__(self, gamma, sigma, k, horizon=None):
        self.g, self.s2, self.k = float(gamma), float(sigma) ** 2, float(k)
        self.h = horizon
        self.name = f'avellaneda_stoikov_g{self.g:g}' + ('' if horizon is None else f'_tau{horizon:g}s')

    def reset(self, ep):
        pass

    def __call__(self, ep, k):
        B, _, A, _ = ep.book()
        mid = (B + A) * 0.5
        tau = (ep.end - ep.t) / DECISION_BINS if self.h is None else self.h
        q = ep.inv / ep.Q
        r = -q * self.g * self.s2 * tau
        half = 0.5 * self.g * self.s2 * tau + math.log1p(self.g / self.k) / self.g
        bps = 1e-4 * mid
        return quote_prices(mid, mid + r * bps, half * bps, B, A)


def estimate_sigma_k(days, max_bps=10.0):
    """sigma: std of 1 s mid log-returns (bps). k: slope of log P(a taker trade reaches
    delta bps beyond mid within the next second) against delta, both sides pooled."""
    rets, ups, dns = [], [], []
    for day in days:
        mid = day.mid_ticks
        idx = np.arange(day.first_bin + DECISION_BINS, day.n_bins - DECISION_BINS, DECISION_BINS)
        m0 = mid[idx - 1]
        rets.append(np.diff(np.log(mid[idx - 1])) * 1e4)
        hi = np.full(len(idx), -np.inf)
        lo = np.full(len(idx), np.inf)
        tb = np.repeat(np.arange(day.n_bins), np.diff(day.tr_start))
        sec = (tb - idx[0]) // DECISION_BINS
        ok = (sec >= 0) & (sec < len(idx))
        bmask = ok & (day.tr_side > 0)
        smask = ok & (day.tr_side < 0)
        np.maximum.at(hi, sec[bmask], day.tr_px[bmask].astype(np.float64))
        np.minimum.at(lo, sec[smask], day.tr_px[smask].astype(np.float64))
        ups.append((hi - m0) / m0 * 1e4)
        dns.append((m0 - lo) / m0 * 1e4)
    r = np.concatenate(rets)
    sigma = float(np.std(r))
    exc = np.concatenate(ups + dns)
    grid = np.linspace(0.0, max_bps, 41)[1:]
    p = np.array([(exc >= d).mean() for d in grid])
    keep = p > 1e-4
    slope, icpt = np.polyfit(grid[keep], np.log(p[keep]), 1)
    return {'sigma_bps_per_sqrt_s': sigma, 'k_per_bps': float(-slope), 'A_per_s': float(math.exp(icpt)),
            'fit_points': int(keep.sum())}


class LearnedActor:
    """v2 actor: observation -> (half-spread, skew) around the microprice."""

    def __init__(self, actors, norm_mean, norm_std, half_max_bps, skew_max_bps, clip=5.0, name='td3_v2'):
        self.actors = actors if isinstance(actors, list) else [actors]
        self.mean = np.asarray(norm_mean, np.float64)
        self.std = np.asarray(norm_std, np.float64)
        self.hmax, self.smax, self.clip = half_max_bps, skew_max_bps, clip
        self.name = name

    def reset(self, ep):
        pass

    def obs(self, ep):
        m = ep.day.features()[ep.t // DECISION_BINS].astype(np.float64)
        x = np.concatenate([m, ep.agent_features()])
        return np.clip((x - self.mean) / self.std, -self.clip, self.clip).astype(np.float32)

    def act(self, z):
        us = [a(z) for a in self.actors]
        return np.mean(us, axis=0) if len(us) > 1 else us[0]

    def __call__(self, ep, k):
        u = self.act(self.obs(ep))
        return action_to_quotes(ep, u, self.hmax, self.smax)


def action_to_quotes(ep, u, hmax, smax):
    B, BQ, A, AQ = ep.book()
    mid = (B + A) * 0.5
    micro = B + (A - B) * BQ / (BQ + AQ)
    u0, u1 = float(u[0]), float(u[1])
    half = hmax * ((u0 + 1.0) * 0.5) ** 2
    skew = smax * u1
    bps = 1e-4 * mid
    return quote_prices(mid, micro + skew * bps, half * bps, B, A)


class NumpyMLP:
    """Float32 MLP (ReLU hidden, tanh out) evaluated in float64 from float32 weights."""

    def __init__(self, layers):
        self.layers = [(np.asarray(L['w'], np.float32).reshape(L['out'], L['in']).astype(np.float64),
                        np.asarray(L['b'], np.float32).astype(np.float64)) for L in layers]

    def __call__(self, z):
        h = np.asarray(z, np.float64)
        for i, (w, b) in enumerate(self.layers):
            h = w @ h + b
            h = np.maximum(h, 0.0) if i < len(self.layers) - 1 else np.tanh(h)
        return h


class PreviousTD3:
    """The v1 actor (trained on LTCUSDT 2024-03-30 in the event-replay env) run through an
    adapter that rebuilds its 52-value observation from the replay: a mirror book of the levels
    the touch has visited (what the v1 converter produced), sizes in lots of 1/10 quote, own
    fills for the trade imbalance, and its 0-10 bps per-side offset map with the repo's
    nearest-tick rounding."""

    FAR = 50.0

    def __init__(self, actor_json_path):
        J = json.loads(open(actor_json_path).read())
        self.net = NumpyMLP(J['layers'])
        n = J['normalizer']
        self.mean, self.std, self.clip = np.asarray(n['mean']), np.asarray(n['std']), n['clip']
        self.half_max = J['action_map']['half_max']
        self.name = 'td3_v1_previous'

    def reset(self, ep):
        self.bids, self.asks = {}, {}
        self.mids = deque(maxlen=100)
        self.depths = deque(maxlen=100)
        self.nfills = 0
        self.entry = 0.0
        self.pos = 0
        self.own = deque(maxlen=10)
        self.last_bin = ep.t - 1

    def _mirror(self, ep):
        day = ep.day
        for b in range(self.last_bin + 1, ep.t):
            for j in range(day.tr_start[b], day.tr_start[b + 1]):
                p = int(day.tr_px[j])
                if day.tr_side[j] > 0:
                    for k in [x for x in self.asks if x < p]:
                        del self.asks[k]
                    for k in [x for x in self.bids if x >= p]:
                        del self.bids[k]
                else:
                    for k in [x for x in self.bids if x > p]:
                        del self.bids[k]
                    for k in [x for x in self.asks if x <= p]:
                        del self.asks[k]
            B, A = int(day.bid[b]), int(day.ask[b])
            for k in [x for x in self.bids if x > B or x >= A]:
                del self.bids[k]
            for k in [x for x in self.asks if x < A or x <= B]:
                del self.asks[k]
            self.bids[B] = int(day.bq[b])
            self.asks[A] = int(day.aq[b])
            self.mids.append((B + A) * 0.5)
            self.depths.append(sum(self.bids.values()) + sum(self.asks.values()))
        self.last_bin = ep.t - 1

    def obs(self, ep):
        self._mirror(ep)
        for f in ep.fills[self.nfills:]:
            q = f[1] * f[3]
            prev = self.pos
            if prev == 0 or prev * q > 0:
                self.entry = (abs(prev) * self.entry + abs(q) * f[2]) / (abs(prev) + abs(q))
            elif abs(q) > abs(prev):
                self.entry = f[2]
            elif abs(q) == abs(prev):
                self.entry = 0.0
            self.pos += q
            self.own.append(q)
        self.nfills = len(ep.fills)
        lot = ep.Q / 10.0
        B, _, A, _ = ep.book()
        mid = (B + A) * 0.5
        x = []
        for book, rev, sign in ((self.bids, True, -1.0), (self.asks, False, 1.0)):
            lv = sorted(book.items(), reverse=rev)[:10]
            for i in range(10):
                if i < len(lv):
                    x += [max(-self.FAR, min(self.FAR, lv[i][0] - mid)), math.log1p(lv[i][1] / lot)]
                else:
                    x += [sign * self.FAR, 0.0]
        tb, ta = sum(self.bids.values()), sum(self.asks.values())
        m = np.asarray(self.mids)
        vol = float(np.std(np.diff(np.log(m)))) if len(m) > 2 else 0.0
        inv = ep.inv / lot
        upnl = inv * (mid - self.entry) * ep.tick if self.pos else 0.0
        live = (ep.bpx is not None) + (ep.apx is not None)
        secs = (ep.t - ep.t0) / DECISION_BINS
        tsf = 1.0 if ep.last_fill_bin is None else min((ep.t - ep.last_fill_bin) / DECISION_BINS / 60.0, 1.0)
        bv = sum(q for q in self.own if q > 0)
        sv = -sum(q for q in self.own if q < 0)
        timb = (bv - sv) / (bv + sv) if bv + sv else 0.0
        d = np.asarray(self.depths, np.float64)
        vflow = float((d[-1] - d.mean()) / d.mean()) if len(d) > 1 and d.mean() else 0.0
        x += [0.0, A - B, (tb - ta) / (tb + ta) if tb + ta else 0.0, vol, inv, upnl,
              inv * mid * ep.tick / (mid * ep.tick), live, (secs % 23400) / 23400, tsf, timb, vflow]
        z = (np.asarray(x) - self.mean) / self.std
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def __call__(self, ep, k):
        u = self.net(self.obs(ep))
        d = self.half_max * (u + 1.0) / 2.0
        B, _, A, _ = ep.book()
        return midpoint_quotes((B + A) * 0.5, float(np.float32(d[0])), float(np.float32(d[1])), B, A)


def run_policy(day, start_bin, policy, record=False, **env_kw):
    """Run one episode; return the per-episode metrics used everywhere in the evaluation."""
    from .sim import Episode, markouts
    ep = Episode(day, start_bin, **env_kw)
    policy.reset(ep)
    eq, invs, steps = [0.0], [], []
    k = 0
    while not ep.done:
        bid, ask = policy(ep, k)
        if record:
            steps.append((ep.t, bid, ask, ep.inv))
        ep.step(bid, ask)
        eq.append(ep.cash + ep.inv * UNIT * ep.mid_price() if not ep.done else ep.cash)
        invs.append(ep.inv / ep.Q)
        k += 1
    eq = np.asarray(eq)
    notional = ep.Q * UNIT * day.mid_ticks[ep.t0] * day.tick
    mo, w = markouts(day, ep.fills, 5 * DECISION_BINS)
    fills = np.asarray(ep.fills, dtype=np.float64).reshape(-1, 4)
    pre = fills[:, 0].astype(np.int64) - 1  # mid at the end of the bin before the fill
    edge = (fills[:, 1] * ((day.bid[pre].astype(np.float64) + day.ask[pre]) * 0.5 - fills[:, 2]) / fills[:, 2] * 1e4
            if len(fills) else np.zeros(0))
    invs = np.asarray(invs)
    out = {
        'symbol': day.symbol, 'day': day.day, 'start_bin': start_bin, 'policy': policy.name,
        'pnl': float(ep.cash), 'steps': k, 'fills': len(ep.fills),
        'filled_units': int(ep.maker_units), 'quote_units': ep.Q, 'quote_notional': float(notional),
        'maker_notional': float(np.sum(fills[:, 3] * UNIT * fills[:, 2] * day.tick)) if len(fills) else 0.0,
        'liq_notional': float(ep.liq_units * UNIT * day.mid_ticks[ep.stop - 1] * day.tick),
        'liquidation_cost': float(ep.liquidation or 0.0),
        'max_drawdown': float(np.max(np.maximum.accumulate(eq) - eq)),
        'mean_abs_inv': float(np.mean(np.abs(invs))) if len(invs) else 0.0,
        'inv_std': float(np.std(invs)) if len(invs) else 0.0,
        'markout_sum': float(np.sum(mo * w)), 'markout_w': float(np.sum(w)),
        'edge_sum': float(np.sum(edge * fills[:, 3])) if len(fills) else 0.0,
    }
    if record:
        out['trace'] = steps
        out['equity'] = eq.tolist()
    return out
