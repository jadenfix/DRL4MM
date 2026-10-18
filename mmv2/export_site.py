"""Export the frozen agent, the per-instrument best baseline, held-out test slices and the
test-set numbers for the browser demo, plus Python reference traces for the TS parity check.

usage: python -m mmv2.export_site DATA RUNS SITE_DATA_DIR PARITY_REF_JSON
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .baselines import make_policy
from .config import SYMBOLS
from .evaluate import LABEL, REQUIRED
from .rl import PolicyCall, feature_mask
from .sim import EPISODE_BINS, Day, Episode, quote_prices  # noqa: F401

SLICE_MINUTES = 5
# fixed before looking at any slice: 14:00 UTC on the first test day of the calm and volatile weeks
SLICE_STARTS = [('2023-08-13', 14), ('2024-03-19', 14)]


def f32(v):
    return [float(repr(np.float32(x)).replace('np.float32(', '').rstrip(')')) for x in np.asarray(v, np.float32).ravel()]


def baseline_spec(fam, name, fit):
    if fam == 'fixed_mid':
        return {'family': fam, 'name': name, 'half_bps': float(name.split('_')[2].replace('bps', ''))}
    if fam == 'repo_fixed_midpoint':
        return {'family': fam, 'name': name, 'bps': 10.0}
    if fam == 'avellaneda_stoikov':
        parts = name.split('_')
        g = float(parts[2][1:])
        h = float(parts[3][3:-1]) if len(parts) > 3 else None
        return {'family': fam, 'name': name, 'gamma': g, 'sigma': fit['sigma_bps_per_sqrt_s'], 'k': fit['k_per_bps'], 'horizon_s': h}
    raise ValueError(fam)


def slice_payload(day, start_bin, stop_bin, latency, max_inv):
    ep = Episode(day, start_bin, latency_bins=latency, stop_bin=stop_bin)
    lo, stop = ep.lo, ep.stop
    ts = day.tr_start[lo:stop + 1].astype(np.int64)
    a, b = int(ts[0]), int(ts[-1])
    F = day.features()[ep.t0 // 10:(stop + 9) // 10]
    return {
        'symbol': day.symbol, 'day': day.day,
        'start_utc': datetime.fromtimestamp((day.meta['day0_ms'] + ep.t0 * 100) / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'tick': day.tick, 'quote_units': day.quote_units, 'max_inv_quotes': max_inv, 'latency_bins': latency,
        'start_bin': start_bin, 't0': ep.t0, 'stop': stop,
        'bid': day.bid[lo:stop].tolist(), 'ask': day.ask[lo:stop].tolist(),
        'bq': day.bq[lo:stop].tolist(), 'aq': day.aq[lo:stop].tolist(),
        'ts': (ts - a).tolist(), 'tside': day.tr_side[a:b].tolist(), 'tpx': day.tr_px[a:b].tolist(), 'tq': day.tr_qty[a:b].tolist(),
        'features': [f32(r) for r in F],
    }


def compact(p):
    """Delta-encode the long integer arrays (decoded in the demo before replay)."""
    q = dict(p)
    for k in ('bid', 'ask', 'tpx'):
        v = np.asarray(p[k], np.int64)
        q[k] = np.r_[v[:1], np.diff(v)].tolist()
    q['ask'] = (np.asarray(p['ask']) - np.asarray(p['bid'])).tolist()
    q['encoding'] = 'bid, tpx: first value then differences; ask: ask - bid'
    return q


def trace(ep_factory, pol):
    ep = ep_factory()
    pol.reset(ep)
    rows = []
    while not ep.done:
        bid, ask = pol(ep, 0)
        de = ep.step(bid, ask)
        rows.append([bid, ask, ep.inv, ep.cash, len(ep.fills), de])
    return {'steps': rows, 'final_cash': ep.cash, 'fills': [list(f) for f in ep.fills], 'liquidation': ep.liquidation}


def main(data, runs, site, ref_path):
    runs, site = Path(runs), Path(site)
    agent = json.loads((runs / 'agent.json').read_text())
    base = json.loads((runs / 'baselines.json').read_text())
    test = json.loads((runs / 'test_results.json').read_text())
    cfg, norm = agent['config'], agent['normalizer']
    site.mkdir(parents=True, exist_ok=True)
    agent_out = {
        'source': 'DRL4MM v2 (branch v2, mmv2/): seed-ensemble actor, mean of tanh outputs',
        'algo': cfg['algo'], 'hidden': cfg['actor_hidden'],
        'normalizer': {'mean': norm['mean'], 'std': norm['std'], 'clip': norm['clip']},
        'mask': feature_mask(cfg['features']).tolist(), 'action': cfg.get('action', 'quadratic'),
        'half_max': cfg['half_max'], 'h0': cfg.get('h0', 7.5), 'skew_max': cfg['skew_max'],
        'anchor': cfg['anchor'],
        'ensemble': [[{'in': L['in'], 'out': L['out'], 'w': f32(L['w']), 'b': f32(L['b'])} for L in net] for net in agent['ensemble']],
    }
    (site / 'v2-agent.json').write_text(json.dumps(agent_out, separators=(',', ':')))
    pol_agent = PolicyCall(agent['ensemble'], norm, cfg, 'agent')
    ref = {'actor_samples': [], 'slices': []}
    index = {'source': 'Binance USD-M futures bookTicker + aggTrades (data.binance.vision), CC BY-NC-SA 4.0',
             'slice_minutes': SLICE_MINUTES, 'instruments': {}}
    rng = np.random.default_rng(0)
    for sym in SYMBOLS:
        e = test['per_symbol'][sym]
        fam = e['best_baseline_by_validation']
        bname = base['chosen'][sym][fam]['policy']
        spec = baseline_spec(fam, bname, base['fits'][sym])
        sa, sb = e['summary']['agent'], e['summary'][fam]
        t = e['vs_best_baseline']
        info = {'baseline': {**spec, 'label': LABEL[fam]}, 'files': [],
                'test': {'episodes': sa['episodes'], 'days': sa['days'],
                         'agent': {k: sa[k] for k in ('pnl_mean', 'pnl_mean_ci', 'sharpe', 'sharpe_ci', 'fills_per_hour', 'mean_abs_inv_quotes', 'markout_5s_bps', 'max_dd_cumulative')},
                         'baseline': {k: sb[k] for k in ('pnl_mean', 'pnl_mean_ci', 'sharpe', 'sharpe_ci', 'fills_per_hour', 'mean_abs_inv_quotes', 'markout_5s_bps', 'max_dd_cumulative')},
                         'diff_mean': t['diff_mean'], 'diff_ci95': t['diff_ci95'], 'p_sign_flip': t['p_sign_flip'],
                         'ex_post': {'label': LABEL[e['best_baseline_ex_post']], 'diff_mean': e['vs_best_baseline_ex_post']['diff_mean'],
                                     'diff_ci95': e['vs_best_baseline_ex_post']['diff_ci95'], 'p_sign_flip': e['vs_best_baseline_ex_post']['p_sign_flip']},
                         'seen_in_training': sym != 'SOLUSDT'}}
        for k, (d, hour) in enumerate(SLICE_STARTS):
            day = Day(Path(data) / f'{sym}_{d}.npz')
            s = hour * EPISODE_BINS
            stop = s + SLICE_MINUTES * 600
            payload = slice_payload(day, s, stop, cfg['latency'], 5)
            name = f'v2-{sym[:3].lower()}-{k}.json'
            (site / name).write_text(json.dumps(compact(payload), separators=(',', ':')))
            info['files'].append({'file': name, 'start_utc': payload['start_utc'], 'week': 'calm' if d < '2024' else 'volatile'})
            fac = lambda day=day, s=s, stop=stop: Episode(day, s, latency_bins=cfg['latency'], stop_bin=stop)  # noqa: E731
            bpol = make_policy(bname, base['fits'][sym], None)
            ra, rb = trace(fac, pol_agent), trace(fac, bpol)
            ref['slices'].append({'file': name, 'agent': ra, 'baseline': rb})
            ep = fac()
            obs = []
            while not ep.done and len(obs) < 40:
                z = pol_agent.inner.obs(ep)
                obs.append(z)
                ep.step(*pol_agent(ep, 0))
            for z in [obs[i] for i in rng.choice(len(obs), 10, replace=False)]:
                u = pol_agent.inner.act(z)
                ref['actor_samples'].append({'z': [float(v) for v in z], 'u': [float(v) for v in u]})
            print(name, (site / name).stat().st_size, 'bytes', 'agent fills', len(ra['fills']), 'baseline fills', len(rb['fills']))
        index['instruments'][sym] = info
    pooled = test['pooled']
    index['pooled'] = {'agent': {k: pooled['summary']['agent'][k] for k in ('pnl_mean', 'pnl_mean_ci', 'sharpe', 'sharpe_ci')},
                       'best_baseline': {k: pooled['summary']['best_baseline'][k] for k in ('pnl_mean', 'pnl_mean_ci', 'sharpe', 'sharpe_ci')},
                       'vs': pooled['vs_best_baseline_calendar_day_clusters']}
    index['required_baselines'] = list(REQUIRED)
    (site / 'v2-index.json').write_text(json.dumps(index, separators=(',', ':')))
    Path(ref_path).write_text(json.dumps(ref))
    for p in sorted(site.glob('v2-*.json')):
        print(p.name, p.stat().st_size)


if __name__ == '__main__':
    main(*sys.argv[1:5])
