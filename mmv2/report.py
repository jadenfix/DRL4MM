"""Stage 6: write results.md (data, baselines, ablations, agent, held-out test) from the
frozen JSON outputs of the earlier stages."""
from __future__ import annotations

import json
from pathlib import Path

from .config import SYMBOLS, days
from .evaluate import LABEL, markdown


def verdict(res):
    lines = []
    for sym, e in res['per_symbol'].items():
        t = e['vs_best_baseline']
        sig = t['diff_ci95'][0] > 0 and t['p_sign_flip'] < 0.05
        worse = t['diff_ci95'][1] < 0 and t['p_sign_flip'] < 0.05
        word = 'beats' if sig else ('is beaten by' if worse else 'is not significantly different from')
        lines.append(f"- {sym}: the agent {word} the best baseline ({LABEL[e['best_baseline_by_validation']]}); "
                     f"difference {t['diff_mean']:+.3f} USDT/h, 95% CI [{t['diff_ci95'][0]:+.3f}, {t['diff_ci95'][1]:+.3f}], "
                     f"sign-flip p = {t['p_sign_flip']:.3f}, HAC p = {t['p_hac']:.3f}.")
    for k, lab in (('vs_best_baseline_calendar_day_clusters', 'calendar-day clusters'),
                   ('vs_best_baseline_instrument_day_clusters', 'instrument-day clusters')):
        t = res['pooled'][k]
        lines.append(f"- Pooled over the four instruments ({lab}): difference {t['diff_mean']:+.3f} USDT/h, "
                     f"95% CI [{t['diff_ci95'][0]:+.3f}, {t['diff_ci95'][1]:+.3f}], sign-flip p = {t['p_sign_flip']:.3f}.")
    return '\n'.join(lines)


def data_section(data_dir):
    m = Path(data_dir) / 'manifest.json'
    if not m.exists():
        return ''
    man = json.loads(m.read_text())
    L = ['| day | week | split | ' + ' | '.join(f'{s} book rows / trades' for s in SYMBOLS) + ' |',
         '|---|---|---|' + '---|' * len(SYMBOLS)]
    for d, week, split in days():
        cells = []
        for s in SYMBOLS:
            e = man.get(f'{s}_{d}')
            cells.append(f"{e['book_rows'] / 1e6:.1f}M / {e['trade_rows'] / 1e6:.2f}M" if e else '-')
        L.append(f'| {d} | {week} | {split} | ' + ' | '.join(cells) + ' |')
    ticks = {s: man[f'{s}_{days()[0][0]}']['tick'] for s in SYMBOLS if f'{s}_{days()[0][0]}' in man}
    L.append('')
    L.append('Tick sizes: ' + ', '.join(f'{s} {t:g}' for s, t in ticks.items()) +
             f". {sum(e['book_rows'] for e in man.values()) / 1e6:.0f}M bookTicker rows and "
             f"{sum(e['trade_rows'] for e in man.values()) / 1e6:.0f}M aggTrades over {len(man)} instrument-days, "
             'every archive verified against its published SHA-256.')
    return '\n'.join(L)


def baseline_section(base):
    L = ['| instrument | sigma (bps/sqrt s) | k (1/bps) | fixed half-spread | Avellaneda-Stoikov | repo 10 bps | previous TD3 | best on validation |',
         '|---|---|---|---|---|---|---|---|']
    req = ('fixed_mid', 'avellaneda_stoikov', 'repo_fixed_midpoint', 'td3_v1_previous')
    for s in SYMBOLS:
        c, f = base['chosen'][s], base['fits'][s]
        best = max(req, key=lambda k: c[k]['pnl_mean'])
        cell = lambda k: f"`{c[k]['policy']}` {c[k]['pnl_mean']:+.2f}"  # noqa: E731
        L.append(f"| {s} | {f['sigma_bps_per_sqrt_s']:.2f} | {f['k_per_bps']:.2f} | {cell('fixed_mid')} | {cell('avellaneda_stoikov')} | "
                 f"{cell('repo_fixed_midpoint')} | {cell('td3_v1_previous')} | {LABEL[best]} |")
    return '\n'.join(L)


def ablation_section(out):
    p = Path(out) / 'ablation_summary.json'
    if not p.exists():
        return ''
    abl = json.loads(p.read_text())
    L = ['| run (1 seed, 400k steps) | settings | val PnL / h (BTC, ETH, LTC pooled) | BTC | ETH | LTC | fills / h | mean abs inv |',
         '|---|---|---|---|---|---|---|---|']
    for name, a in abl.items():
        cfg = {k: v for k, v in a['config'].items() if k not in ('seed', 'steps')}
        ps = a['per_symbol']
        L.append(f"| {name} | `{json.dumps(cfg)}` | {a['val_pnl_mean']:+.2f} | {ps['BTCUSDT']:+.2f} | {ps['ETHUSDT']:+.2f} | "
                 f"{ps['LTCUSDT']:+.2f} | {a['fills_per_hour']:.1f} | {a['mean_abs_inv']:.2f} |")
    return '\n'.join(L)


def write_report(out, data_dir=None, log=print):
    out = Path(out)
    res = json.loads((out / 'test_results.json').read_text())
    base = json.loads((out / 'baselines.json').read_text())
    agent = json.loads((out / 'agent.json').read_text())
    cfg = {k: agent['config'][k] for k in ('algo', 'reward', 'eta', 'gamma', 'anchor', 'features', 'action', 'h0', 'half_max',
                                           'skew_max', 'latency', 'actor_hidden', 'critic_hidden', 'steps') if k in agent['config']}
    md = ['# DRL4MM v2 results', '',
          '## Verdict on the held-out test days', '', verdict(res), '',
          '## Held-out test (each instrument: 6 days, 144 hourly episodes)', '', markdown(res, base, agent), '',
          'PnL is USDT per hourly episode for ~1,000 USDT quotes, after liquidating at the touch; CIs are 95% day-block bootstrap. '
          'Sharpe-like = mean / std of hourly PnL. Markout: 5 s mid move after a fill, signed for the agent (negative = adverse selection). '
          'Edge at fill: fill price vs the mid just before the fill. "VIP0 fees" subtracts 2 bps maker / 5 bps taker post hoc.', '',
          '## Agent', '', f"Final configuration: `{json.dumps(cfg)}`", '',
          'Members (checkpoint picked on the validation subset during training): ' +
          ', '.join(f"seed {m['seed']} @ {m['checkpoint_steps']} steps" for m in agent['members']), '',
          'Full-validation mean PnL / h: ' + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in agent['validation_pnl_mean'].items()}), '',
          '## Ablations (validation days only)', '', ablation_section(out), '',
          '## Baselines (sigma, k fitted on training days; parameters chosen on validation; validation mean PnL / h)', '',
          baseline_section(base), '']
    if data_dir:
        md += ['## Data', '', data_section(data_dir), '']
    md += [f"Frozen inputs (SHA-256): `{json.dumps(res.get('frozen_inputs_sha256', {}))}`", '']
    (out / 'results.md').write_text('\n'.join(md))
    log('\n'.join(md))
