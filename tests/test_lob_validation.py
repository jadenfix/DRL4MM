"""Regression tests for atomic rejection and FIFO order-book integrity."""

from dataclasses import replace

import numpy as np
import pytest

from simulator.lob_simulator import LimitOrderBook, LOBSimulator, Order, OrderSide, OrderType


def resting_order():
    """Return an order that can remain on the book."""
    return Order("first", "AAPL", OrderSide.SELL, 101.0, 10, 0.0)


@pytest.mark.parametrize("changes", [
    {"order_id": ""}, {"symbol": "MSFT"}, {"side": "buy"},
    {"order_type": OrderType.CANCEL}, {"quantity": 0}, {"quantity": -1},
    {"quantity": 1.5}, {"quantity": True}, {"price": 0.0},
    {"price": float("nan")}, {"price": float("inf")},
    {"timestamp": float("nan")}, {"ttl": 0.0}, {"ttl": -1.0},
    {"ttl": float("inf")},
])
def test_invalid_crossing_order_leaves_book_unchanged(changes):
    """Invalid aggressive orders must not consume liquidity or add corrupt levels."""
    book = LimitOrderBook("AAPL")
    book.add_order(resting_order())
    incoming = Order("incoming", "AAPL", OrderSide.BUY, 102.0, 5, 1.0)
    with pytest.raises(ValueError):
        book.add_order(replace(incoming, **changes))
    assert book.orders["first"].quantity == 10
    assert book.asks[101.0].total_quantity == 10
    assert book.trade_count == 0
    assert list(book.orders) == ["first"]


def test_duplicate_id_rejected_before_matching_and_cancellation_remains_consistent():
    """Duplicate IDs previously overwrote tracking while leaving old FIFO entries."""
    book = LimitOrderBook("AAPL")
    book.add_order(resting_order())
    with pytest.raises(ValueError, match="Duplicate"):
        book.add_order(Order("first", "AAPL", OrderSide.BUY, 102.0, 5, 1.0))
    assert book.cancel_order("first")
    assert not book.orders and not book.asks and not book.ask_prices
    assert book.trade_count == 0


def test_fifo_partial_fills_and_numpy_integer_quantities():
    """Validation preserves market sweeps and numerical integer quantities."""
    book = LimitOrderBook("AAPL")
    book.add_order(resting_order())
    book.add_order(Order("second", "AAPL", OrderSide.SELL, 101.0, 10, 0.5))
    trades = book.add_order(Order("sweep", "AAPL", OrderSide.BUY, 0.0,
                                  np.int64(15), 1.0, OrderType.MARKET))
    assert [(t.seller_id, t.quantity) for t in trades] == [("first", 10), ("second", 5)]
    assert book.asks[101.0].total_quantity == 5
    assert book.cancel_order("second")
    assert not book.asks and not book.orders


def test_agent_side_typo_rejected_without_order():
    """A mistyped buy side must not silently become a sell order."""
    simulator = LOBSimulator("AAPL")
    with pytest.raises(ValueError, match="side"):
        simulator.place_agent_order("byu", 100.0, 10)
    assert not simulator.lob.orders
    assert simulator.order_id_counter == 0
