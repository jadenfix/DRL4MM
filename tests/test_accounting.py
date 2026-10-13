"""Accounting invariants using executed LOB orders, not market-performance data."""
import numpy as np
import pytest

from simulator.lob_simulator import Order, OrderSide
from simulator.market_env import MarketMakingEnv


def execute_fill(env, side, price, quantity, identity):
    """Cross an actual agent limit order against an isolated counterparty order."""
    counterparty = OrderSide.SELL if side == 'buy' else OrderSide.BUY
    env.simulator.lob.add_order(Order(
        order_id=f'counterparty_{identity}', symbol='AAPL', side=counterparty,
        price=price, quantity=quantity, timestamp=float(identity),
    ))
    env.simulator.place_agent_order(side, price, quantity)
    env._update_agent_state()


def assert_reconciled(env):
    assert env.cash + env.position_value == pytest.approx(
        env.realized_pnl + env.unrealized_pnl
    )


@pytest.mark.parametrize('first,second', [('buy', 'sell'), ('sell', 'buy')])
def test_round_trip_cash_and_pnl(first, second):
    env = MarketMakingEnv()
    env.simulator.latency_mean = 0
    entry, exit_price = (100.0, 105.0) if first == 'buy' else (105.0, 100.0)
    execute_fill(env, first, entry, 25, 1)
    assert env.realized_pnl == 0
    assert env.unrealized_pnl == 0
    assert_reconciled(env)
    execute_fill(env, second, exit_price, 25, 2)
    assert env.inventory == 0
    assert env.cash == pytest.approx(125)
    assert env.realized_pnl == pytest.approx(125)
    assert env.unrealized_pnl == 0
    assert env.fill_count == 2
    assert_reconciled(env)


def test_display_history_rollover_does_not_recount_or_lose_inventory():
    env = MarketMakingEnv()
    env.simulator.latency_mean = 0
    for identity in range(1005):
        execute_fill(env, 'buy', 100.0, 1, identity)
    assert len(env.recent_fills) == 100
    assert env.inventory == env.fill_count == env.episode_trades == 1005
    assert env.cash == -100500
    assert env.realized_pnl == env.unrealized_pnl == 0
    assert_reconciled(env)
    env._update_agent_state()
    assert env.inventory == env.fill_count == 1005
    env.inventory_penalty = 0
    assert env._calculate_reward() == pytest.approx(1005 * env.config['fill_bonus'])
    assert env._calculate_reward() == 0
    env.reset(seed=1)
    assert env.inventory == env.cash == env.fill_count == env._processed_trade_count == 0


def test_self_trade_does_not_create_inventory_profit_or_fill_rewards():
    env = MarketMakingEnv()
    env.simulator.latency_mean = 0
    env.simulator.place_agent_order('sell', 100.0, 1)
    env.simulator.place_agent_order('buy', 100.0, 1)
    env._update_agent_state()
    assert env.inventory == env.cash == env.fill_count == 0
    assert env.realized_pnl == env.unrealized_pnl == 0
    assert env._calculate_reward() == 0


def test_partial_close_and_position_reversal_preserve_average_cost():
    env = MarketMakingEnv()
    env.simulator.latency_mean = 0
    execute_fill(env, 'buy', 100.0, 10, 1)
    execute_fill(env, 'buy', 110.0, 10, 2)
    assert env._entry_price == 105
    execute_fill(env, 'sell', 120.0, 5, 3)
    assert env.inventory == 15
    assert env.realized_pnl == 75
    assert_reconciled(env)
    execute_fill(env, 'sell', 120.0, 20, 4)
    assert env.inventory == -5
    assert env._entry_price == 120
    assert env.realized_pnl == 300
    assert env.unrealized_pnl == 0
    assert_reconciled(env)


def test_missing_retained_trades_fail_instead_of_fabricating_pnl():
    env = MarketMakingEnv()
    execute_fill(env, 'buy', 100.0, 1, 1)
    env.simulator.lob.trade_count += 1001
    with pytest.raises(RuntimeError, match='history gap'):
        env._update_agent_state()


def test_step_accounts_for_immediate_quote_fill():
    env = MarketMakingEnv()
    env.simulator.latency_mean = 0
    env.simulator.lob.add_order(Order('outside_bid', 'AAPL', OrderSide.BUY, 99.0, 100, 0.0))
    env.simulator.lob.add_order(Order('outside_ask', 'AAPL', OrderSide.SELL, 101.0, 100, 0.0))
    observation, reward, done, info = env.step(np.array([0.02, 0.05]))
    assert info['fill_count'] == 1
    assert info['inventory'] == 100
    assert np.isfinite(observation).all() and np.isfinite(reward)
    assert_reconciled(env)
