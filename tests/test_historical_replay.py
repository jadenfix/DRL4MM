"""Chronological software fixtures, not measured or fabricated market history."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from simulator.historical import load_events
from simulator.market_env import MarketMakingEnv
from simulator.replay import run_replay


def write_fixture(path, rows=None):
    """Explicit tiny order-event fixture exercising the real matching path."""
    rows = rows if rows is not None else [
        [1700000000, 'add', 'bid', 'buy', 99, 20],
        [1700000000, 'add', 'ask', 'sell', 101, 20],
        [1700000001, 'market', 'sell_flow', 'sell', '', 1],
        [1700000002, 'market', 'buy_flow', 'buy', '', 1],
        [1700000003, 'add', 'next_bid', 'buy', 98, 20],
    ]
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['timestamp', 'event', 'order_id', 'side', 'price', 'quantity'])
        writer.writerows(rows)
    return path


def test_real_cli_exports_reconciled_replay(tmp_path):
    source = write_fixture(tmp_path / 'fixture.csv')
    output = tmp_path / 'result.json'
    process = subprocess.run([
        sys.executable, '-m', 'simulator.replay', str(source), '--output', str(output),
    ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    result = json.loads(output.read_text())
    assert result['complete'] is True
    assert result['events_consumed'] == result['events_total'] == 5
    assert len(result['rows']) == 3
    assert len(result['fills']) == 2
    assert [f['side'] for f in result['fills']] == ['buy', 'sell']
    assert result['final']['inventory'] == 0
    assert result['final']['cash'] == pytest.approx(0.2)
    assert result['final']['equity'] == pytest.approx(0.2)
    for row in result['rows']:
        assert row['equity'] == pytest.approx(row['realized_pnl'] + row['unrealized_pnl'])
    assert len(result['input_sha256']) == 64
    assert run_replay(source)['final'] == result['final']


def test_reset_primes_only_initial_book_and_first_action_precedes_next_event(tmp_path):
    source = write_fixture(tmp_path / 'fixture.csv')
    env = MarketMakingEnv(data_source='historical', config={
        'historical_data_path': source, 'quote_size': 1, 'latency_mean': 0,
    })
    observation = env.reset()
    assert np.isfinite(observation).all()
    assert env._historical_cursor == 2
    assert env.current_time == 1700000000
    assert env.fill_count == 0
    observation, _, done, info = env.step(np.array([-0.001, 0.001], dtype=np.float32))
    assert not done  # Epoch timestamp must not be mistaken for episode length.
    assert info['inventory'] == info['fill_count'] == info['episode_steps'] == 1
    assert info['timestamp'] == 1700000001
    assert info['active_quotes'] == observation[env.lob_depth * 4 + 7] == 1
    while not done:
        _, _, done, _ = env.step(np.array([-0.001, 0.001], dtype=np.float32))
    with pytest.raises(RuntimeError, match='episode ended'):
        env.step(np.zeros(2, dtype=np.float32))
    env.reset()
    assert env._historical_cursor == 2 and env.fill_count == 0


@pytest.mark.parametrize('replacement', [
    [1699999999, 'add', 'next', 'buy', 98, 20],
    [float('nan'), 'add', 'next', 'buy', 98, 20],
    [1700000001, 'add', 'next', 'buy', 0, 20],
    [1700000001, 'add', 'next', 'buy', 98, -1],
    [1700000001, 'add', 'next', 'buy', 98, '1.5'],
    [1700000001, 'add', 'next', 'hold', 98, 20],
    [1700000001, 'unknown', 'next', 'buy', 98, 20],
])
def test_malformed_events_fail_before_replay(tmp_path, replacement):
    source = write_fixture(tmp_path / 'fixture.csv')
    rows = list(csv.reader(source.open()))
    write_fixture(source, rows[1:3] + [replacement])
    with pytest.raises(ValueError, match='Invalid replay row'):
        load_events(source)


def test_cancellation_and_duplicate_order_ids(tmp_path):
    source = write_fixture(tmp_path / 'fixture.csv', [
        [0, 'add', 'bid', 'buy', 99, 20], [0, 'add', 'ask', 'sell', 101, 20],
        [1, 'add', 'extra', 'buy', 98, 10], [2, 'cancel', 'extra', '', '', ''],
        [3, 'add', 'extra', 'buy', 98, 10],
    ])
    assert run_replay(source)['complete']
    write_fixture(source, [
        [0, 'add', 'bid', 'buy', 99, 20], [0, 'add', 'ask', 'sell', 101, 20],
        [1, 'add', 'bid', 'buy', 100, 10],
    ])
    with pytest.raises(ValueError, match='Duplicate active'):
        run_replay(source)


def test_partial_run_is_explicit_and_cli_refuses_input_overwrite(tmp_path):
    source = write_fixture(tmp_path / 'fixture.csv')
    result = run_replay(source, max_steps=1)
    assert not result['complete'] and result['events_consumed'] == 3
    original = source.read_bytes()
    process = subprocess.run([
        sys.executable, '-m', 'simulator.replay', str(source), '--output', str(source),
    ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert process.returncode == 2
    assert source.read_bytes() == original
    output = tmp_path / 'partial.json'
    partial = subprocess.run([
        sys.executable, '-m', 'simulator.replay', str(source), '--output', str(output),
        '--max-steps', '1',
    ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert partial.returncode == 2
    assert json.loads(output.read_text())['complete'] is False


def test_missing_initial_book_and_unknown_cancellation_fail(tmp_path):
    source = write_fixture(tmp_path / 'fixture.csv', [[0, 'add', 'bid', 'buy', 99, 20]])
    with pytest.raises(ValueError, match='two-sided'):
        run_replay(source)
    source = write_fixture(tmp_path / 'fixture.csv', [
        [0, 'add', 'bid', 'buy', 99, 20], [0, 'add', 'ask', 'sell', 101, 20],
        [1, 'cancel', 'missing', '', '', ''],
    ])
    with pytest.raises(ValueError, match='inactive external'):
        run_replay(source)


def test_external_agent_like_ids_do_not_count_as_agent_fills(tmp_path):
    source = write_fixture(tmp_path / 'fixture.csv', [
        [0, 'add', 'bid', 'buy', 99, 20], [0, 'add', 'ask', 'sell', 101, 20],
        [1, 'market', 'agent_fake', 'sell', '', 1],
    ])
    result = run_replay(source, quote_bps=100)
    assert result['final']['fill_count'] == 0


def test_actions_and_event_limits_are_validated(tmp_path):
    source = write_fixture(tmp_path / 'fixture.csv')
    with pytest.raises(ValueError, match='exceeds 2 events'):
        load_events(source, max_events=2)
    env = MarketMakingEnv(data_source='historical', config={'historical_data_path': source})
    env.reset()
    for action in ([np.nan, 0], [0], [0.2, 0]):
        with pytest.raises(ValueError, match='Action'):
            env.step(action)
        assert env._historical_cursor == 2


def test_training_and_evaluation_factories_select_supplied_historical_data(tmp_path):
    pytest.importorskip('torch')
    import yaml
    from train.train_td3 import create_environment as training_environment
    from train.evaluate import create_environment as evaluation_environment
    source = write_fixture(tmp_path / 'fixture.csv')
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/td3_config.yaml').read_text())
    config['data']['historical_data_path'] = str(source)
    config['environment']['quote_size'] = 1
    config['simulation']['latency_mean'] = 0
    for factory in (training_environment, evaluation_environment):
        env = factory(config)
        assert env.data_source == 'historical'
        assert np.isfinite(env.reset()).all()
        _, _, done, info = env.step(np.array([-0.001, 0.001], dtype=np.float32))
        assert not done and info['inventory'] == 1


def test_unimplemented_live_adapter_fails_explicitly():
    with pytest.raises(NotImplementedError, match='Live exchange data'):
        MarketMakingEnv(data_source='live').reset()
