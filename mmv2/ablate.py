"""Stage 3 (design selection, validation only): train one seed per candidate configuration,
then score each run's best checkpoint on the full validation days (BTC/ETH/LTC)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .rl import DEFAULTS, evaluate, train

ABLATIONS = {
    'td3_full': {},
    'sac_full': {'algo': 'sac'},
    'td3_basic_features': {'features': 'basic'},
    'td3_inv_penalty': {'eta': 0.002},
    'td3_mid_anchor': {'anchor': 'mid'},
    'td3_gamma995': {'gamma': 0.995},
    'td3_markout_reward': {'reward': 'markout', 'eta': 0.002},
    'sac_markout_reward': {'algo': 'sac', 'reward': 'markout', 'eta': 0.002},
    'td3_residual_markout': {'action': 'residual', 'reward': 'markout', 'eta': 0.002},
    'sac_residual_markout': {'algo': 'sac', 'action': 'residual', 'reward': 'markout', 'eta': 0.002},
    'td3_residual_pnl': {'action': 'residual', 'eta': 0.002},
    # round 2, on top of SAC + markout reward (the best of round 1 on validation)
    'sac_markout_mid_anchor': {'algo': 'sac', 'reward': 'markout', 'eta': 0.002, 'anchor': 'mid'},
    'sac_markout_basic_features': {'algo': 'sac', 'reward': 'markout', 'eta': 0.002, 'features': 'basic'},
    'sac_markout_eta01': {'algo': 'sac', 'reward': 'markout', 'eta': 0.01},
}


def score(run_dir, data):
    d = Path(run_dir)
    cfg = {**DEFAULTS, **json.loads((d / 'config.json').read_text())}
    norm = json.loads((d / 'normalizer.json').read_text())
    layers = json.loads((d / 'actor_best.json').read_text())
    res = evaluate([layers], norm, cfg, data, 'val')
    p = np.array([r['pnl'] for r in res])
    return {'val_pnl_mean': float(p.mean()), 'val_pnl_std': float(p.std()), 'episodes': len(p),
            'per_symbol': {s: float(np.mean([r['pnl'] for r in res if r['symbol'] == s])) for s in sorted({r['symbol'] for r in res})},
            'fills_per_hour': float(np.mean([r['fills'] for r in res])),
            'mean_abs_inv': float(np.mean([r['mean_abs_inv'] for r in res]))}


def run(data, out, names, steps, log=print):
    out = Path(out)
    summary_path = out / 'ablation_summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in names:
        if name in summary:
            continue
        cfg = {**ABLATIONS[name], 'seed': 0, 'steps': steps}
        train(cfg, data, out / f'abl_{name}', log=lambda m, n=name: log(f'[{n}] {m}'))
        summary[name] = {'config': cfg, **score(out / f'abl_{name}', data)}
        summary_path.write_text(json.dumps(summary, indent=1))
        log(f'[{name}] full validation: ' + json.dumps(summary[name]))
    return summary


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], sys.argv[4].split(',') if len(sys.argv) > 4 else list(ABLATIONS), int(sys.argv[3]),
        log=lambda m: print(m, flush=True))
