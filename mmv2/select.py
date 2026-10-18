"""Stage 4: freeze the agent. Each seed contributes the checkpoint that scored best on the
validation subset during training; the shipped agent averages their actions (seed ensemble).
The full validation days are re-run for the ensemble and each member (BTC/ETH/LTC only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .rl import DEFAULTS, evaluate


def select(data, out, runs, log=print):
    out = Path(out)
    members, cfg, norm, info = [], None, None, []
    for r in runs:
        d = out / r
        c = {**DEFAULTS, **json.loads((d / 'config.json').read_text())}
        n = json.loads((d / 'normalizer.json').read_text())
        if cfg is None:
            cfg, norm = c, n
        keys = ('features', 'anchor', 'half_max', 'skew_max', 'latency', 'actor_hidden')
        assert all(c[k] == cfg[k] for k in keys) and n == norm, f'{r}: runs differ in a policy-defining setting'
        prog = json.loads((d / 'progress.json').read_text())
        best = max(prog, key=lambda p: p['val_pnl_mean'])
        members.append(json.loads((d / 'actor_best.json').read_text()))
        info.append({'run': r, 'seed': c['seed'], 'checkpoint_steps': best['steps'], 'val_subset_pnl': best['val_pnl_mean']})
    val = {}
    for i, m in enumerate(members):
        res = evaluate([m], norm, cfg, data, 'val')
        val[f'seed{info[i]["seed"]}'] = float(np.mean([x['pnl'] for x in res]))
    res = evaluate(members, norm, cfg, data, 'val')
    val['ensemble'] = float(np.mean([x['pnl'] for x in res]))
    val['ensemble_per_symbol'] = {s: float(np.mean([x['pnl'] for x in res if x['symbol'] == s]))
                                  for s in sorted({x['symbol'] for x in res})}
    agent = {'config': cfg, 'normalizer': norm, 'ensemble': members, 'members': info, 'validation_pnl_mean': val}
    (out / 'agent.json').write_text(json.dumps(agent))
    log('selected ' + json.dumps({'members': info, 'validation': val}))
    return agent
