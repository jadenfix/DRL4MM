"""Synthetic instrument-days in the converted NPZ format (software fixtures for tests only;
not market data)."""
from __future__ import annotations

import json

import numpy as np

from .data import BIN_MS, BINS_PER_DAY


def write_day(path, bid, spread, bq, aq, trades, tick=0.1, symbol='TESTUSDT', day='2000-01-01', upd=None):
    """bid/spread in ticks, bq/aq in 0.001 units (arrays of length n_bins <= one day);
    trades: list of (bin, side, price_ticks, units)."""
    n = len(bid)
    full = lambda x, v: np.r_[np.asarray(x), np.full(BINS_PER_DAY - n, v)]  # noqa: E731
    bid, spread = full(bid, bid[-1]), full(spread, spread[-1])
    bq, aq = full(bq, bq[-1]), full(aq, aq[-1])
    tr = sorted(trades)
    base = int(bid[0])
    meta = {'symbol': symbol, 'day': day, 'day0_ms': 0, 'bin_ms': BIN_MS, 'tick': tick, 'base_ticks': base,
            'first_book_bin': 0, 'license': 'synthetic fixture'}
    np.savez(path, meta=json.dumps(meta), bid=(bid - base).astype(np.int32), spread=spread.astype(np.int32),
             bid_qty=bq.astype(np.int32), ask_qty=aq.astype(np.int32),
             upd=(np.ones(BINS_PER_DAY) if upd is None else full(upd, 0)).astype(np.uint8),
             tr_bin=np.array([t[0] for t in tr], np.int32), tr_side=np.array([t[1] for t in tr], np.int8),
             tr_px=np.array([t[2] - base for t in tr], np.int32), tr_qty=np.array([t[3] for t in tr], np.int32),
             tr_n=np.ones(len(tr), np.uint8))


def random_day(path, n_bins=40000, seed=0, tick=0.1, start=300000):
    """A random-walk touch with Poisson-ish trades at the touch and occasional sweeps."""
    rng = np.random.default_rng(seed)
    bid = start + np.cumsum(rng.choice([-1, 0, 0, 0, 0, 0, 1], n_bins))
    spread = rng.choice([1, 1, 1, 1, 2], n_bins)
    bq = rng.integers(1, 40000, n_bins)
    aq = rng.integers(1, 40000, n_bins)
    trades = []
    for b in range(1, n_bins):
        if rng.random() < 0.3:
            side = int(rng.choice([-1, 1]))
            px = int(bid[b - 1]) if side < 0 else int(bid[b - 1] + spread[b - 1])
            if rng.random() < 0.05:
                px += side * int(rng.integers(1, 4))
            trades.append((b, side, px, int(rng.integers(1, 30000))))
    write_day(path, bid, spread, bq, aq, trades, tick=tick)
    return bid, spread, bq, aq, trades
