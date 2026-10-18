"""Queue-position replay of one instrument-day for a single passive market maker.

Time runs on the 100 ms bins of the converted data. The agent decides once per second
(``DECISION_BINS`` bins) using only state up to the end of the previous bin; with
``latency_bins`` = L its new quotes reach the book L bins later (the old quotes stay live in
between). Within each bin the recorded trades are matched against the agent's resting orders
first, then the book moves to its recorded end-of-bin state.

Fill rules for a resting bid at price p (asks mirror them):

* a seller-initiated trade below p means the taker swept through p: the order fills in full;
* seller-initiated volume exactly at p first consumes ``queue_ahead``; the excess fills us;
* if the recorded best ask moves to <= p, an incoming sell would have crossed us: full fill;
* queue bookkeeping when the bin ends: best bid == p -> queue_ahead = min(queue_ahead, size)
  (cancellations are assumed to come from behind us first; a level we were placed behind
  becomes visible with everything displayed ahead of us); best bid < p -> we are alone at the
  best price (queue_ahead = 0); best bid > p -> unknown queue, unchanged.

Orders are placed post-only: a bid at or above the current best ask is rejected. A new order
joins the back of its level: queue_ahead = displayed size at the touch, 0 inside the spread,
unknown (-1) behind the touch. Re-sending an unchanged price keeps the order and its queue
place. A side is not quoted when a full fill could push |inventory| past the limit.

Quantities are integer units of 0.001 base asset and prices integer ticks, so the matching is
exact integer arithmetic. The replayed market does not react to the agent (no impact).
Cash, inventory and every fill are recorded; at the end of the episode the remaining
inventory is liquidated at the touch (crossing the spread) and that is the episode PnL.
"""
from __future__ import annotations

import json
import math

import numpy as np

from .config import DECISION_BINS, EPISODE_SECONDS, MAX_INV_QUOTES, QUOTE_NOTIONAL

UNIT = 0.001
LOT_UNITS = {'SOLUSDT': 1000}  # exchange quantity step in units (SOLUSDT trades whole coins)
EPISODE_BINS = EPISODE_SECONDS * DECISION_BINS
N_MARKET = 16


class Day:
    """One converted instrument-day, integer prices/quantities, plus causal market features."""

    def __init__(self, path):
        z = np.load(path)
        self.meta = meta = json.loads(str(z['meta']))
        self.symbol, self.day = meta['symbol'], meta['day']
        self.tick = float(meta['tick'])
        base = meta['base_ticks']
        # int32 storage keeps a day at ~30 MB; Episode converts its hour to Python ints
        self.bid = (z['bid'] + np.int32(base)).astype(np.int32)
        self.ask = (self.bid + z['spread']).astype(np.int32)
        self.bq = z['bid_qty']
        self.aq = z['ask_qty']
        self.upd = z['upd']
        self.n_bins = len(self.bid)
        self.tr_start = np.searchsorted(z['tr_bin'], np.arange(self.n_bins + 1)).astype(np.int32)
        self.tr_side = z['tr_side']
        self.tr_px = (z['tr_px'] + np.int32(base)).astype(np.int32)
        self.tr_qty = z['tr_qty']
        self.first_bin = int(meta['first_book_bin'])
        mid0 = (int(self.bid[self.first_bin]) + int(self.ask[self.first_bin])) * 0.5 * self.tick
        lot = LOT_UNITS.get(self.symbol, 1)
        self.quote_units = max(lot, int(round(QUOTE_NOTIONAL / mid0 / UNIT / lot)) * lot)
        self._features = None

    @property
    def mid_ticks(self):
        return (self.bid.astype(np.float64) + self.ask) * 0.5

    def features(self):
        """float32 [n_decisions, N_MARKET]; row m describes the book at the end of bin 10m-1
        and uses only bins <= 10m-1 (row 0 is zeros: nothing observed yet)."""
        if self._features is None:
            self._features = market_features(self)
            self.upd = None
        return self._features

    def hours(self):
        """Episode start bins (clock hours) with a usable book."""
        return [h * EPISODE_BINS for h in range(self.n_bins // EPISODE_BINS)
                if h * EPISODE_BINS + EPISODE_BINS > self.first_bin + 20 * DECISION_BINS]


def _window_sum(x, w):
    """s[j] = sum x[max(0, j-w+1) .. j] (trailing, causal)."""
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    j = np.arange(len(x))
    return c[j + 1] - c[np.maximum(0, j - w + 1)]


def market_features(day):
    bid, ask = day.bid.astype(np.float64), day.ask.astype(np.float64)
    bq, aq = day.bq.astype(np.float64), day.aq.astype(np.float64)
    n = len(bid)
    mid = (bid + ask) * 0.5
    spread = ask - bid
    depth = (bq + aq) * 0.5
    depth_avg = np.maximum(_window_sum(depth, 600) / np.minimum(np.arange(n) + 1, 600), 1.0)
    # Cont-Kukanov-Stoikov order-flow imbalance between consecutive bins
    pb, pa, pbq, paq = np.r_[bid[0], bid[:-1]], np.r_[ask[0], ask[:-1]], np.r_[bq[0], bq[:-1]], np.r_[aq[0], aq[:-1]]
    e = (bid >= pb) * bq - (bid <= pb) * pbq - (ask <= pa) * aq + (ask >= pa) * paq
    e[0] = 0.0
    buy = np.zeros(n)
    sell = np.zeros(n)
    tb = np.repeat(np.arange(n), np.diff(day.tr_start))
    q = day.tr_qty.astype(np.float64)
    np.add.at(buy, tb[day.tr_side > 0], q[day.tr_side > 0])
    np.add.at(sell, tb[day.tr_side < 0], q[day.tr_side < 0])
    lm = np.log(mid * day.tick)
    r2 = np.r_[0.0, np.diff(lm) ** 2]

    def lag(x, k):
        return np.r_[np.full(k, x[0]), x[:-k]]

    def tfi(w):
        b, s = _window_sum(buy, w), _window_sum(sell, w)
        return np.where(b + s > 0, (b - s) / np.maximum(b + s, 1e-12), 0.0)

    F = np.stack([
        np.minimum(spread, 20.0),                                  # 0 spread in ticks
        day.tick / (mid * day.tick) * 1e4,                         # 1 tick size in bps
        (bq - aq) / np.maximum(bq + aq, 1.0),                      # 2 queue imbalance at the touch
        spread / mid * 1e4,                                        # 3 spread in bps
        np.clip(_window_sum(e, 10) / depth_avg, -10, 10),          # 4 OFI 1 s / mean touch depth
        np.clip(_window_sum(e, 50) / depth_avg, -10, 10),          # 5 OFI 5 s
        tfi(10),                                                   # 6 trade-flow imbalance 1 s
        tfi(100),                                                  # 7 trade-flow imbalance 10 s
        np.log1p(_window_sum(buy + sell, 100) / depth_avg),        # 8 traded volume 10 s / depth
        np.sqrt(_window_sum(r2, 100)) * 1e4,                       # 9 realized vol 10 s (bps)
        np.sqrt(_window_sum(r2, 600)) * 1e4,                       # 10 realized vol 60 s (bps)
        (lm - lag(lm, 10)) * 1e4,                                  # 11 mid return 1 s (bps)
        (lm - lag(lm, 100)) * 1e4,                                 # 12 mid return 10 s
        (lm - lag(lm, 600)) * 1e4,                                 # 13 mid return 60 s
        np.log1p(_window_sum(day.upd.astype(np.float64), 10)),     # 14 book updates 1 s
        np.log(np.maximum(depth, 1.0) / depth_avg),                # 15 touch depth vs 60 s mean
    ], axis=1)
    out = np.zeros((n // DECISION_BINS, N_MARKET), np.float32)
    out[1:] = F[DECISION_BINS - 1:n - 1:DECISION_BINS][:len(out) - 1]
    return out


class Episode:
    """One clock hour of one day. ``step(bid_ticks, ask_ticks)`` (None = no quote on that side)."""

    def __init__(self, day: Day, start_bin: int, latency_bins=1, max_inv_quotes=MAX_INV_QUOTES,
                 maker_fee_bps=0.0, taker_fee_bps=0.0, quote_units=None, stop_bin=None):
        """``stop_bin`` ends (and liquidates) the episode early, e.g. for a short replay slice;
        features and policies still see the clock hour's end as the horizon."""
        assert 0 <= latency_bins < DECISION_BINS
        self.day = day
        self.L = latency_bins
        self.Q = quote_units or day.quote_units
        self.cap = max_inv_quotes * self.Q
        self.maker_fee = maker_fee_bps * 1e-4
        self.taker_fee = taker_fee_bps * 1e-4
        self.tick = day.tick
        s = start_bin
        self.end = s + EPISODE_BINS
        first_dec = -(-(day.first_bin + 1) // DECISION_BINS) * DECISION_BINS
        self.t0 = self.t = max(s, first_dec)
        self.stop = self.end if stop_bin is None else min(self.end, stop_bin)
        lo = self.t0 - 1
        self.lo = lo
        sl = slice(lo, self.stop)
        self.B, self.A = day.bid[sl].tolist(), day.ask[sl].tolist()
        self.BQ, self.AQ = day.bq[sl].tolist(), day.aq[sl].tolist()
        ts = day.tr_start[lo:self.stop + 1]
        self.ts = (ts - ts[0]).tolist()
        tsl = slice(int(ts[0]), int(ts[-1]))
        self.tside, self.tpx, self.tq = day.tr_side[tsl].tolist(), day.tr_px[tsl].tolist(), day.tr_qty[tsl].tolist()
        self.inv = 0
        self.cash = 0.0
        # resting orders: price (None = none), remaining units, queue ahead (-1 = unknown)
        self.bpx = self.apx = None
        self.brem = self.arem = 0
        self.bqa = self.aqa = -1
        self.pending = None
        self.fills = []           # (bin, side +1 buy/-1 sell, price ticks, units)
        self.last_fill_bin = None
        self.maker_units = 0
        self.liq_units = 0
        self.done = self.t >= self.stop
        self.liquidation = None

    # --- state at decision time (end of bin t-1)
    def book(self):
        i = self.t - 1 - self.lo
        return self.B[i], self.BQ[i], self.A[i], self.AQ[i]

    def mid_price(self):
        b, _, a, _ = self.book()
        return (b + a) * 0.5 * self.tick

    def equity(self):
        return self.cash + self.inv * UNIT * self.mid_price()

    # --- order handling
    def _fill(self, side, b, units):
        px = self.bpx if side > 0 else self.apx
        notional = units * UNIT * px * self.tick
        if side > 0:
            self.inv += units
            self.cash -= notional
            self.brem -= units
            if self.brem == 0:
                self.bpx, self.bqa = None, -1
        else:
            self.inv -= units
            self.cash += notional
            self.arem -= units
            if self.arem == 0:
                self.apx, self.aqa = None, -1
        self.cash -= notional * self.maker_fee
        self.maker_units += units
        self.fills.append((b, side, px, units))
        self.last_fill_bin = b

    def _activate(self, b):
        bid_t, ask_t = self.pending
        self.pending = None
        i = b - 1 - self.lo
        B, BQ, A, AQ = self.B[i], self.BQ[i], self.A[i], self.AQ[i]
        if bid_t is not None and bid_t == self.bpx:
            if self.inv + self.brem > self.cap:
                self.bpx, self.brem, self.bqa = None, 0, -1
        elif bid_t is None or bid_t >= A or self.inv + self.Q > self.cap:
            self.bpx, self.brem, self.bqa = None, 0, -1
        else:
            self.bpx, self.brem = bid_t, self.Q
            self.bqa = 0 if bid_t > B else (BQ if bid_t == B else -1)
        if ask_t is not None and ask_t == self.apx:
            if self.inv - self.arem < -self.cap:
                self.apx, self.arem, self.aqa = None, 0, -1
        elif ask_t is None or ask_t <= B or self.inv - self.Q < -self.cap:
            self.apx, self.arem, self.aqa = None, 0, -1
        else:
            self.apx, self.arem = ask_t, self.Q
            self.aqa = 0 if ask_t < A else (AQ if ask_t == A else -1)

    def step(self, bid_ticks, ask_ticks):
        """Run one decision interval. Returns the equity change marked to mid (price units)."""
        if self.done:
            raise RuntimeError('episode finished')
        e0 = self.equity()
        self.pending = (bid_ticks, ask_ticks)
        t = self.t
        act = t + self.L
        stop = min(t + DECISION_BINS, self.stop)
        ts, tside, tpx, tq = self.ts, self.tside, self.tpx, self.tq
        for b in range(t, stop):
            if b == act:
                self._activate(b)
            i = b - self.lo
            j0, j1 = ts[i], ts[i + 1]
            if j1 > j0 and (self.bpx is not None or self.apx is not None):
                bvol = avol = 0
                bsweep = asweep = False
                p, q = self.bpx, self.apx
                for j in range(j0, j1):
                    if tside[j] < 0:
                        if p is not None:
                            if tpx[j] < p:
                                bsweep = True
                            elif tpx[j] == p:
                                bvol += tq[j]
                    elif q is not None:
                        if tpx[j] > q:
                            asweep = True
                        elif tpx[j] == q:
                            avol += tq[j]
                if p is not None:
                    if bsweep:
                        self._fill(1, b, self.brem)
                    elif bvol > 0 and self.bqa >= 0:
                        if bvol > self.bqa:
                            self._fill(1, b, min(self.brem, bvol - self.bqa))
                            if self.bpx is not None:
                                self.bqa = 0
                        else:
                            self.bqa -= bvol
                if q is not None:
                    if asweep:
                        self._fill(-1, b, self.arem)
                    elif avol > 0 and self.aqa >= 0:
                        if avol > self.aqa:
                            self._fill(-1, b, min(self.arem, avol - self.aqa))
                            if self.apx is not None:
                                self.aqa = 0
                        else:
                            self.aqa -= avol
            B, A = self.B[i], self.A[i]
            p = self.bpx
            if p is not None:
                if A <= p:
                    self._fill(1, b, self.brem)
                elif B == p:
                    self.bqa = self.BQ[i] if self.bqa < 0 else min(self.bqa, self.BQ[i])
                elif B < p:
                    self.bqa = 0
            q = self.apx
            if q is not None:
                if B >= q:
                    self._fill(-1, b, self.arem)
                elif A == q:
                    self.aqa = self.AQ[i] if self.aqa < 0 else min(self.aqa, self.AQ[i])
                elif A > q:
                    self.aqa = 0
        self.t = stop
        if self.pending is not None:  # latency longer than the remaining episode
            self.pending = None
        de = self.equity() - e0
        if self.t >= self.stop:
            self.done = True
            self._liquidate()
        return de

    def _liquidate(self):
        b, _, a, _ = self.book()
        before = self.equity()
        if self.inv > 0:
            notional = self.inv * UNIT * b * self.tick
            self.cash += notional - notional * self.taker_fee
        elif self.inv < 0:
            notional = -self.inv * UNIT * a * self.tick
            self.cash -= notional + notional * self.taker_fee
        self.liq_units = abs(self.inv)
        self.liquidation = self.cash - before  # cost of crossing (<= 0), price units
        self.inv = 0
        self.bpx = self.apx = None

    # --- agent-state features (appended to the market features for learned policies)
    def agent_features(self):
        B, BQ, A, AQ = self.book()
        dep = max((BQ + AQ) * 0.5, 1.0)
        secs_since = ((self.t - (self.last_fill_bin if self.last_fill_bin is not None else self.t0))
                      / DECISION_BINS)
        out = [self.inv / self.Q, math.log1p(min(secs_since, 600.0)), (self.end - self.t) / EPISODE_BINS]
        for px, qa, sgn, ref in ((self.bpx, self.bqa, 1, B), (self.apx, self.aqa, -1, A)):
            if px is None:
                out += [0.0, 0.0, 5.0]
            else:
                out += [1.0, max(-2.0, min(10.0, float((ref - px) * sgn))), 5.0 if qa < 0 else min(5.0, qa / dep)]
        return out


N_AGENT = 9
N_OBS = N_MARKET + N_AGENT


def quote_prices(mid_t, center_t, half_t, B, A):
    """Integer quote prices around a center (all in ticks); post-only against the current touch."""
    bid = math.floor(center_t - half_t - 1e-9)
    ask = math.ceil(center_t + half_t + 1e-9)
    return min(bid, A - 1), max(ask, B + 1)


def markouts(day, fills, horizon_bins):
    """Per-fill signed markout in bps: (mid after horizon - fill price) x side / price."""
    if not fills:
        return np.zeros(0), np.zeros(0)
    f = np.asarray(fills, dtype=np.int64)
    later = np.minimum(f[:, 0] + horizon_bins, day.n_bins - 1)
    mid = (day.bid[later].astype(np.float64) + day.ask[later]) * 0.5
    return f[:, 1] * (mid - f[:, 2]) / f[:, 2] * 1e4, f[:, 3].astype(np.float64)
