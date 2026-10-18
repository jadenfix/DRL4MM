"""Episode metrics, day-block bootstrap intervals and paired tests."""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats as sps

VIP0 = {'maker_bps': 2.0, 'taker_bps': 5.0}  # Binance USD-M regular-user fees, for sensitivity only


def _by_day(rows, key):
    days = sorted({(r['symbol'], r['day']) for r in rows})
    return [np.array([key(r) for r in rows if (r['symbol'], r['day']) == d]) for d in days]


def block_bootstrap(groups, stat, n=10000, seed=0):
    """Resample whole groups (days) with replacement; returns (point, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    point = stat(np.concatenate(groups))
    k = len(groups)
    vals = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, k, k)
        vals[i] = stat(np.concatenate([groups[j] for j in pick]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def sharpe(x):
    x = np.asarray(x, np.float64)
    s = x.std()
    return float(x.mean() / s) if s > 0 else 0.0


def chrono(rows):
    return sorted(rows, key=lambda r: (r['day'], r['symbol'], r['start_bin']))


def summarize(rows, boot=True, seed=0):
    p = np.array([r['pnl'] for r in rows])
    cum = np.cumsum([r['pnl'] for r in chrono(rows)])
    w = sum(r['markout_w'] for r in rows)
    out = {
        'episodes': len(rows), 'days': len({(r['symbol'], r['day']) for r in rows}),
        'pnl_mean': float(p.mean()), 'pnl_total': float(p.sum()), 'pnl_std': float(p.std()),
        'sharpe': sharpe(p), 'win_rate': float((p > 0).mean()),
        'max_dd_episode_mean': float(np.mean([r['max_drawdown'] for r in rows])),
        'max_dd_cumulative': float(np.max(np.maximum.accumulate(np.r_[0, cum]) - np.r_[0, cum])),
        'mean_abs_inv_quotes': float(np.mean([r['mean_abs_inv'] for r in rows])),
        'inv_std_quotes': float(np.mean([r['inv_std'] for r in rows])),
        'fills_per_hour': float(np.mean([r['fills'] for r in rows])),
        'filled_notional_per_hour': float(np.mean([r['maker_notional'] for r in rows])),
        'markout_5s_bps': float(sum(r['markout_sum'] for r in rows) / w) if w else float('nan'),
        'edge_at_fill_bps': float(sum(r['edge_sum'] for r in rows) / w) if w else float('nan'),
        'liquidation_cost_mean': float(np.mean([r['liquidation_cost'] for r in rows])),
        'pnl_mean_vip0_fees': float(np.mean([r['pnl'] - VIP0['maker_bps'] * 1e-4 * r['maker_notional']
                                             - VIP0['taker_bps'] * 1e-4 * r['liq_notional'] for r in rows])),
    }
    if boot:
        g = _by_day(rows, lambda r: r['pnl'])
        out['pnl_mean_ci'] = block_bootstrap(g, np.mean, seed=seed)[1:]
        out['sharpe_ci'] = block_bootstrap(g, sharpe, seed=seed)[1:]
    return out


def newey_west_t(d, lag):
    d = np.asarray(d, np.float64)
    n = len(d)
    e = d - d.mean()
    s = e @ e / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    se = math.sqrt(max(s, 1e-300) / n)
    t = d.mean() / se
    return float(t), float(2 * sps.t.sf(abs(t), n - 1))


def sign_flip_by_cluster(diff_by_cluster, max_exact=20, n_mc=200000, seed=0):
    """Two-sided paired sign-flip test on cluster means (exact when <= 2^max_exact flips)."""
    m = np.array([c.mean() for c in diff_by_cluster])
    obs = abs(m.mean())
    k = len(m)
    if k <= max_exact:
        signs = np.array(list(itertools.product([-1, 1], repeat=k)))
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_mc, k))
    null = np.abs((signs * m).mean(1))
    return float((null >= obs - 1e-12).mean())


def paired(rows_a, rows_b, cluster='instrument-day', seed=0):
    """Agent (a) minus baseline (b), matched episode by episode."""
    key = lambda r: (r['symbol'], r['day'], r['start_bin'])  # noqa: E731
    b = {key(r): r for r in rows_b}
    pairs = [(r, b[key(r)]) for r in chrono(rows_a) if key(r) in b]
    d = np.array([x['pnl'] - y['pnl'] for x, y in pairs])
    ckey = (lambda r: (r['symbol'], r['day'])) if cluster == 'instrument-day' else (lambda r: r['day'])
    clusters = sorted({ckey(x) for x, _ in pairs})
    groups = [np.array([x['pnl'] - y['pnl'] for x, y in pairs if ckey(x) == c]) for c in clusters]
    pt, lo, hi = block_bootstrap(groups, np.mean, seed=seed)
    t, p_hac = newey_west_t(d, lag=6)
    return {'n_episodes': len(d), 'n_clusters': len(groups), 'cluster': cluster,
            'diff_mean': pt, 'diff_ci95': [lo, hi], 'p_sign_flip': sign_flip_by_cluster(groups),
            'hac_t': t, 'p_hac': p_hac, 'episodes_better': float((d > 0).mean())}
