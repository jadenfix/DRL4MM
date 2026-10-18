"""Fill rules and accounting of the v2 queue-position replay (synthetic software fixtures)."""
import numpy as np
import pytest

from mmv2.sim import Day, Episode, UNIT
from mmv2.synth import write_day

N = 36000


def make(tmp_path, trades=(), bid=None, bq=None, spread=None, aq=None):
    bid = np.full(N, 1000) if bid is None else bid
    spread = np.ones(N, int) if spread is None else spread
    bq = np.full(N, 5000) if bq is None else bq
    aq = np.full(N, 5000) if aq is None else aq
    p = tmp_path / 'd.npz'
    write_day(p, bid, spread, bq, aq, list(trades))
    return Day(p)


def episode(day, L=0, cap=5):
    return Episode(day, 0, latency_bins=L, quote_units=1000, max_inv_quotes=cap)


def test_queue_depletion_then_fills(tmp_path):
    day = make(tmp_path, [(12, -1, 1000, 3000), (13, -1, 1000, 2500), (14, -1, 1000, 100)])
    ep = episode(day)
    assert ep.t == 10
    ep.step(1000, None)
    assert [f[3] for f in ep.fills] == [500, 100]
    assert ep.inv == 600 and ep.bqa == 0 and ep.brem == 400
    assert ep.cash == pytest.approx(-600 * UNIT * 1000 * 0.1)


def test_sweep_and_cross_fill_in_full(tmp_path):
    day = make(tmp_path, [(12, -1, 999, 1)])
    ep = episode(day)
    ep.step(1000, None)
    assert ep.inv == 1000 and ep.bpx is None
    ask = np.full(N, 1001)
    ask[15:] = 1000  # recorded ask moves onto our resting bid
    bid = np.full(N, 1000)
    bid[15:] = 999
    day = make(tmp_path, bid=bid, spread=ask - bid)
    ep = episode(day)
    ep.step(1000, None)
    assert ep.inv == 1000


def test_cancellations_ahead_and_queue_kept_on_unchanged_price(tmp_path):
    bq = np.full(N, 5000)
    bq[12:] = 1200
    day = make(tmp_path, [(25, -1, 1000, 1500)], bq=bq)
    ep = episode(day)
    ep.step(1000, None)
    assert ep.bqa == 1200
    ep.step(1000, None)  # same price: keeps place (1200 ahead), so 300 fills
    assert ep.inv == 300
    ep2 = episode(day)
    ep2.step(1000, None)
    ep2.step(None, None)
    ep2.step(1000, None)  # re-joined at the back behind 1200 at t=30; no trades after
    assert ep2.inv == 0 and ep2.bqa == 1200


def test_post_only_inside_spread_and_behind_touch(tmp_path):
    spread = np.full(N, 3)
    day = make(tmp_path, [(12, -1, 1001, 50)], spread=spread)
    ep = episode(day)
    ep.step(1001, 1003)  # bid inside the spread: first in queue; ask joins the best ask
    assert ep.inv == 50 and ep.apx == 1003
    ep = episode(day)
    ep.step(1003, 1000)  # both cross the recorded touch: rejected
    assert ep.bpx is None and ep.apx is None
    bid = np.full(N, 1000)
    bid[20:] = 999
    bq = np.full(N, 5000)
    bq[20:] = 300
    day = make(tmp_path, [(15, -1, 999, 10_000), (25, -1, 999, 400)], bid=bid, bq=bq)
    ep = episode(day)
    ep.step(999, None)
    assert ep.bqa == -1 and ep.inv == 0  # behind the touch: unknown queue, no fill at 999 yet
    ep.step(999, None)
    assert ep.inv == 100  # level became the touch with 300 ahead; 400 traded


def test_latency_inventory_cap_and_liquidation(tmp_path):
    day = make(tmp_path, [(10, -1, 1000, 6000), (12, -1, 1000, 6000), (30, -1, 1000, 6000)])
    ep = episode(day, L=1, cap=1)
    ep.step(1000, None)  # live from bin 11: the bin-10 trade is missed
    assert ep.inv == 1000
    ep.step(1000, None)  # cap = 1 quote: no new bid
    assert ep.bpx is None and ep.inv == 1000
    with pytest.raises(AssertionError):
        Episode(day, 0, latency_bins=10)
    ep = episode(day, cap=1)
    while not ep.done:
        ep.step(1000, 1001)
    assert ep.inv == 0
    # bought 1000 units at 1000, liquidated at the 1000 bid: zero PnL, no ask fills recorded
    assert ep.cash == pytest.approx(0.0, abs=1e-9)
    assert ep.liquidation == pytest.approx(-1000 * UNIT * 0.5 * 0.1)


def test_market_features_shape(tmp_path):
    day = make(tmp_path)
    F = day.features()
    assert F.shape == (86400, 16) and np.all(F[0] == 0)
