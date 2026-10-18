"""Stage 2: fit (training days) and tune (validation days) the baselines, per instrument.

* fixed symmetric half-spread around the mid, grid ``HALF_GRID`` bps -> best on validation;
* Avellaneda-Stoikov: sigma and k fitted on the instrument's training days; gamma and the
  horizon (time left in the episode, or a constant 30/120/600 s) tuned on validation;
* the repo's fixed-midpoint policy (10 bps, simulator/replay.py default), untuned;
* the previous (v1) TD3 actor, untuned;
* extra, for attribution: fixed half-spread around the microprice (same grid).

Selection criterion everywhere: mean PnL per validation episode (USDT). SOL baselines are
fitted and tuned on SOL's own train/validation days, which the RL agent never sees.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from .config import SYMBOLS, split_days
from .policies import AvellanedaStoikov, FixedSpread, PreviousTD3, RepoFixedMidpoint, estimate_sigma_k, run_policy
from .sim import Day

HALF_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]
GAMMA_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
HORIZONS = [None, 30.0, 120.0, 600.0]  # None: time left in the episode (classic AS)


def _path(data, sym, d):
    return Path(data) / f'{sym}_{d}.npz'


def fit_as(data, sym):
    return estimate_sigma_k(Day(_path(data, sym, d)) for d in split_days('train'))  # one day at a time


def candidate_policies(fit, v1_actor):
    pols = [FixedSpread(h) for h in HALF_GRID]
    pols += [FixedSpread(h, 'micro') for h in HALF_GRID]
    pols += [AvellanedaStoikov(g, fit['sigma_bps_per_sqrt_s'], fit['k_per_bps'], h) for h in HORIZONS for g in GAMMA_GRID]
    pols += [RepoFixedMidpoint(10.0)]
    if v1_actor:
        pols.append(PreviousTD3(v1_actor))
    return pols


def _run_day(args):
    data, sym, d, fit, v1_actor, latency, pol_names = args
    day = Day(_path(data, sym, d))
    rows = []
    for pol in candidate_policies(fit, v1_actor):
        if pol_names is not None and pol.name not in pol_names:
            continue
        for s in day.hours():
            rows.append(run_policy(day, s, pol, latency_bins=latency))
    return rows


def run_grid(data, split, fits, v1_actor, latency=1, symbols=SYMBOLS, pol_names=None, workers=3):
    tasks = [(str(data), sym, d, fits[sym], v1_actor, latency, pol_names.get(sym) if pol_names else None)
             for sym in symbols for d in split_days(split)]
    with mp.get_context('fork').Pool(workers) as pool:
        out = pool.map(_run_day, tasks, chunksize=1)
    return [r for rows in out for r in rows]


def family(name):
    for f in ('fixed_mid', 'fixed_micro', 'avellaneda_stoikov', 'repo_fixed_midpoint', 'td3_v1_previous'):
        if name.startswith(f):
            return f
    return name


def tune(data, out_dir, v1_actor, latency=1, log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fits = {}
    for sym in SYMBOLS:
        fits[sym] = fit_as(data, sym)
        log(f'{sym} AS fit on training days: {fits[sym]}')
    rows = run_grid(data, 'val', fits, v1_actor, latency)
    (out / 'val_episodes.json').write_text(json.dumps(rows))
    table, chosen = {}, {}
    for sym in SYMBOLS:
        table[sym] = {}
        for name in sorted({r['policy'] for r in rows}):
            p = np.array([r['pnl'] for r in rows if r['symbol'] == sym and r['policy'] == name])
            table[sym][name] = {'pnl_mean': float(p.mean()), 'pnl_std': float(p.std()), 'n': len(p),
                                'fills': float(np.mean([r['fills'] for r in rows if r['symbol'] == sym and r['policy'] == name]))}
        chosen[sym] = {}
        for fam in ('fixed_mid', 'fixed_micro', 'avellaneda_stoikov', 'repo_fixed_midpoint', 'td3_v1_previous'):
            cands = {k: v for k, v in table[sym].items() if family(k) == fam}
            if cands:
                best = max(cands, key=lambda k: cands[k]['pnl_mean'])
                chosen[sym][fam] = {'policy': best, **cands[best]}
        log(sym + ' ' + json.dumps({f: (c['policy'], round(c['pnl_mean'], 4)) for f, c in chosen[sym].items()}))
    res = {'fits': fits, 'val_table': table, 'chosen': chosen, 'latency_bins': latency,
           'half_grid_bps': HALF_GRID, 'gamma_grid': GAMMA_GRID, 'as_horizons_s': HORIZONS}
    (out / 'baselines.json').write_text(json.dumps(res, indent=1))
    return res


def make_policy(name, fit, v1_actor):
    for p in candidate_policies(fit, v1_actor):
        if p.name == name:
            return p
    raise KeyError(name)
