"""Stage 6: write results.md from the frozen test results."""
from __future__ import annotations

import json
from pathlib import Path

from .evaluate import markdown


def verdict(res):
    lines = []
    for sym, e in res['per_symbol'].items():
        t = e['vs_best_baseline']
        sig = t['diff_ci95'][0] > 0 and t['p_sign_flip'] < 0.05
        worse = t['diff_ci95'][1] < 0 and t['p_sign_flip'] < 0.05
        word = 'beats' if sig else ('is beaten by' if worse else 'is not significantly different from')
        lines.append(f"- {sym}: the agent {word} the best baseline ({e['best_baseline_by_validation']}); "
                     f"difference {t['diff_mean']:+.3f} USDT/h, 95% CI [{t['diff_ci95'][0]:+.3f}, {t['diff_ci95'][1]:+.3f}], "
                     f"sign-flip p = {t['p_sign_flip']:.3f}.")
    t = res['pooled']['vs_best_baseline_calendar_day_clusters']
    lines.append(f"- Pooled (clustered by calendar day): difference {t['diff_mean']:+.3f} USDT/h, "
                 f"95% CI [{t['diff_ci95'][0]:+.3f}, {t['diff_ci95'][1]:+.3f}], sign-flip p = {t['p_sign_flip']:.3f}.")
    return '\n'.join(lines)


def write_report(out, log=print):
    out = Path(out)
    res = json.loads((out / 'test_results.json').read_text())
    base = json.loads((out / 'baselines.json').read_text())
    agent = json.loads((out / 'agent.json').read_text())
    md = ['# DRL4MM v2 held-out test results', '', verdict(res), '', markdown(res, base, agent), '']
    (out / 'results.md').write_text('\n'.join(md))
    log('\n'.join(md))
