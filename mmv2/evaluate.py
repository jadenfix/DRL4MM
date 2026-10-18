"""Stage 5: the single held-out test evaluation and the results report.

Everything evaluated here was frozen beforehand: the baselines' parameters (baselines.json,
tuned on validation), the agent configuration and checkpoints (selected on validation).
``evaluate_test`` refuses to run twice into the same directory unless forced, and records the
SHA-256 of everything it evaluated.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from pathlib import Path


from .baselines import make_policy
from .config import SYMBOLS, split_days
from .rl import PolicyCall
from .sim import Day
from .policies import run_policy
from .stats import paired, summarize

REQUIRED = ('fixed_mid', 'avellaneda_stoikov', 'repo_fixed_midpoint', 'td3_v1_previous')
LABEL = {'fixed_mid': 'best fixed half-spread (mid)', 'avellaneda_stoikov': 'Avellaneda-Stoikov',
         'repo_fixed_midpoint': 'repo fixed-midpoint 10 bps', 'td3_v1_previous': 'previous TD3 (v1)',
         'fixed_micro': 'fixed half-spread around microprice (extra)', 'agent': 'v2 agent (seed ensemble)'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _day_task(args):
    data, sym, d, agent, base, v1, split = args
    day = Day(Path(data) / f'{sym}_{d}.npz')
    pols = [PolicyCall(agent['ensemble'], agent['normalizer'], agent['config'], 'agent')]
    pols += [PolicyCall([L], agent['normalizer'], agent['config'], f'agent_seed{i}') for i, L in enumerate(agent['ensemble'])]
    for fam, c in base['chosen'][sym].items():
        p = make_policy(c['policy'], base['fits'][sym], v1)
        p.name = fam
        pols.append(p)
    rows = []
    for pol in pols:
        for s in day.hours():
            r = run_policy(day, s, pol, latency_bins=agent['config']['latency'])
            r['policy'] = pol.name
            rows.append(r)
    return rows


def run_split(data, agent, base, v1, split, workers=3, symbols=SYMBOLS):
    tasks = [(str(data), sym, d, agent, base, v1, split) for sym in symbols for d in split_days(split)]
    with mp.get_context('fork').Pool(workers) as pool:
        out = pool.map(_day_task, tasks, chunksize=1)
    return [r for rows in out for r in rows]


def report(rows, base, split):
    """Per instrument and pooled summaries; paired tests vs the validation-chosen best baseline."""
    res = {'split': split, 'per_symbol': {}, 'pooled': {}}
    policies = sorted({r['policy'] for r in rows})
    for sym in SYMBOLS + ['ALL']:
        sel = [r for r in rows if sym == 'ALL' or r['symbol'] == sym]
        if not sel:
            continue
        summ = {p: summarize([r for r in sel if r['policy'] == p]) for p in policies}
        entry = {'summary': summ}
        if sym != 'ALL':
            val = base['chosen'][sym]
            best = max(REQUIRED, key=lambda f: val[f]['pnl_mean'])
            entry['best_baseline_by_validation'] = best
            entry['vs_best_baseline'] = paired([r for r in sel if r['policy'] == 'agent'],
                                               [r for r in sel if r['policy'] == best], 'instrument-day')
            ex_post = max(REQUIRED, key=lambda f: summ[f]['pnl_mean'])
            entry['best_baseline_ex_post'] = ex_post
            entry['vs_best_baseline_ex_post'] = paired([r for r in sel if r['policy'] == 'agent'],
                                                       [r for r in sel if r['policy'] == ex_post], 'instrument-day')
            entry['vs_each'] = {f: paired([r for r in sel if r['policy'] == 'agent'], [r for r in sel if r['policy'] == f])
                                for f in REQUIRED + ('fixed_micro',)}
            res['per_symbol'][sym] = entry
        else:
            # pooled comparison: each instrument against its own validation-chosen best baseline
            a, b = [], []
            for s in SYMBOLS:
                best = max(REQUIRED, key=lambda f: base['chosen'][s][f]['pnl_mean'])
                a += [r for r in rows if r['symbol'] == s and r['policy'] == 'agent']
                b += [r for r in rows if r['symbol'] == s and r['policy'] == best]
            entry['vs_best_baseline_calendar_day_clusters'] = paired(a, b, 'calendar-day')
            entry['vs_best_baseline_instrument_day_clusters'] = paired(a, b, 'instrument-day')
            bb = []
            for s in SYMBOLS:
                best = max(REQUIRED, key=lambda f: base['chosen'][s][f]['pnl_mean'])
                bb += [dict(r, policy='best_baseline') for r in rows if r['symbol'] == s and r['policy'] == best]
            entry['summary']['best_baseline'] = summarize(bb)
            res['pooled'] = entry
    return res


def evaluate_test(data, agent_path, base_path, v1, out_dir, force=False, workers=3):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'test_episodes.json'
    if target.exists() and not force:
        raise SystemExit(f'{target} exists: the test set has already been evaluated (use --force to overwrite)')
    agent = json.loads(Path(agent_path).read_text())
    base = json.loads(Path(base_path).read_text())
    frozen = {'agent': sha(agent_path), 'baselines': sha(base_path), 'v1_actor': sha(v1)}
    rows = run_split(data, agent, base, v1, 'test', workers)
    target.write_text(json.dumps({'frozen_inputs_sha256': frozen, 'rows': rows}))
    res = report(rows, base, 'test')
    res['frozen_inputs_sha256'] = frozen
    (out / 'test_results.json').write_text(json.dumps(res, indent=1, default=float))
    return res


def fmt_ci(ci, d=3):
    return f'[{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]'


def markdown(res, base, agent_meta):
    L = []
    L.append('| instrument | policy | mean PnL / h (USDT) [95% CI] | Sharpe-like [95% CI] | max DD (cum.) | mean abs inv (quotes) | fills / h | 5 s markout (bps) | edge at fill (bps) | PnL / h at VIP0 fees |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    order = ['agent', 'fixed_mid', 'avellaneda_stoikov', 'repo_fixed_midpoint', 'td3_v1_previous', 'fixed_micro']
    for sym in SYMBOLS + ['ALL']:
        e = res['pooled'] if sym == 'ALL' else res['per_symbol'][sym]
        for p in order + (['best_baseline'] if sym == 'ALL' else []):
            s = e['summary'].get(p)
            if not s:
                continue
            name = LABEL.get(p, p)
            if sym != 'ALL' and p in base['chosen'][sym]:
                name += f" `{base['chosen'][sym][p]['policy']}`"
            if p == 'best_baseline':
                name = 'best baseline (per instrument, chosen on validation)'
            L.append(f"| {sym} | {name} | {s['pnl_mean']:+.3f} {fmt_ci(s['pnl_mean_ci'])} | {s['sharpe']:+.3f} {fmt_ci(s['sharpe_ci'], 2)} | "
                     f"{s['max_dd_cumulative']:.2f} | {s['mean_abs_inv_quotes']:.2f} | {s['fills_per_hour']:.1f} | "
                     f"{s['markout_5s_bps']:+.2f} | {s['edge_at_fill_bps']:+.2f} | {s['pnl_mean_vip0_fees']:+.3f} |")
    L.append('')
    L.append('| instrument | best baseline (validation) | agent - baseline, mean PnL / h [95% day-block CI] | sign-flip p (days) | HAC p | episodes better |')
    L.append('|---|---|---|---|---|---|')
    for sym in SYMBOLS:
        e = res['per_symbol'][sym]
        t = e['vs_best_baseline']
        L.append(f"| {sym} | {LABEL[e['best_baseline_by_validation']]} | {t['diff_mean']:+.3f} {fmt_ci(t['diff_ci95'])} | {t['p_sign_flip']:.3f} | {t['p_hac']:.3f} | {t['episodes_better']:.2f} |")
    for k, lab in (('vs_best_baseline_calendar_day_clusters', 'pooled, clustered by calendar day'),
                   ('vs_best_baseline_instrument_day_clusters', 'pooled, clustered by instrument-day')):
        t = res['pooled'][k]
        L.append(f"| ALL | {lab} | {t['diff_mean']:+.3f} {fmt_ci(t['diff_ci95'])} | {t['p_sign_flip']:.3f} | {t['p_hac']:.3f} | {t['episodes_better']:.2f} |")
    return '\n'.join(L)
