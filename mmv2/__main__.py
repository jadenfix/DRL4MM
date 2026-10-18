"""DRL4MM v2 pipeline: one entry point from public data to a results report.

    python -m mmv2 all --data DATA_DIR --out RUNS_DIR          # everything below, in order
    python -m mmv2 fetch     --data DATA_DIR                    # download + verify + convert
    python -m mmv2 baselines --data DATA_DIR --out RUNS_DIR     # fit on train, tune on validation
    python -m mmv2 ablate    --data DATA_DIR --out RUNS_DIR     # design ablations, validation only
    python -m mmv2 train     --data DATA_DIR --out RUNS_DIR --name td3_s0 --set seed=0 ...
    python -m mmv2 select    --data DATA_DIR --out RUNS_DIR --runs td3_s0,td3_s1,...
    python -m mmv2 test      --data DATA_DIR --out RUNS_DIR     # touches the test days, once
    python -m mmv2 report    --out RUNS_DIR                     # results.md from test_results.json

Resource use: one process trains at a time on 4 torch threads; evaluation uses 3 worker
processes; data is converted one instrument-day at a time without writing raw archives.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')

FINAL = json.loads((Path(__file__).parent / 'final_config.json').read_text()) \
    if (Path(__file__).parent / 'final_config.json').exists() else {}


def parse_set(items):
    cfg = {}
    for it in items or []:
        k, v = it.split('=', 1)
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    return cfg


def v1_actor_path():
    return str(Path(__file__).parent / 'assets' / 'td3_v1_actor.json')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m mmv2', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['fetch', 'baselines', 'ablate', 'train', 'select', 'test', 'report', 'all'])
    ap.add_argument('--data', default='data/v2')
    ap.add_argument('--out', default='runs/v2')
    ap.add_argument('--name')
    ap.add_argument('--set', nargs='*')
    ap.add_argument('--runs')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    out = Path(a.out)
    log = lambda m: print(m, flush=True)  # noqa: E731

    if a.stage in ('fetch', 'all'):
        from .fetch import fetch_all
        fetch_all(a.data, log=log)
    if a.stage in ('baselines', 'all'):
        from .baselines import tune
        tune(a.data, out, v1_actor_path(), log=log)
    if a.stage == 'ablate':  # design selection on validation days (not part of 'all')
        from .ablate import ABLATIONS, run
        run(a.data, out, a.runs.split(',') if a.runs else list(ABLATIONS), int(parse_set(a.set).get('steps', 400000)), log=log)
    if a.stage == 'train':
        from .rl import train
        train(parse_set(a.set), a.data, out / a.name, log=log)
    if a.stage == 'all':
        from .rl import train
        for seed in FINAL['seeds']:
            name = f"final_s{seed}"
            if not (out / name / 'progress.json').exists():
                train({**FINAL['config'], 'seed': seed}, a.data, out / name, log=log)
    if a.stage in ('select', 'all'):
        from .select import select
        runs = a.runs.split(',') if a.runs else [f'final_s{s}' for s in FINAL['seeds']]
        select(a.data, out, runs, log=log)
    if a.stage in ('test', 'all'):
        from .evaluate import evaluate_test
        evaluate_test(a.data, out / 'agent.json', out / 'baselines.json', v1_actor_path(), out, force=a.force)
    if a.stage in ('report', 'all'):
        from .report import write_report
        write_report(out, a.data, log=log)


if __name__ == '__main__':
    main()
