"""Public CLI checks using tiny software events, not historical market evidence."""
import csv
import logging
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from train.train_td3 import training_schedule, create_agent, train_episode
from train.evaluate import create_environment, load_agent
from utils.metrics import MetricsTracker

ROOT = Path(__file__).resolve().parents[1]


def write_config(tmp_path, *, total=8, horizon=23400, save=2, override=False):
    """Use the existing chronological test scenario and real small TD3 networks."""
    source = tmp_path / 'software-events.csv'
    with source.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['timestamp', 'event', 'order_id', 'side', 'price', 'quantity'])
        writer.writerows([
            [1700000000, 'add', 'bid', 'buy', 99, 20],
            [1700000000, 'add', 'ask', 'sell', 101, 20],
            [1700000001, 'market', 'sell_flow', 'sell', '', 1],
            [1700000002, 'market', 'buy_flow', 'buy', '', 1],
            [1700000003, 'add', 'next_bid', 'buy', 98, 20],
        ])
    config = yaml.safe_load((ROOT / 'config/td3_config.yaml').read_text())
    config['data']['historical_data_path'] = str(source)
    config['environment'].update(max_episode_steps=horizon, quote_size=1)
    config['simulation'].update(latency_mean=0, latency_std=0)
    config['agent'].update(hidden_dims=[8, 8], buffer_size=32, batch_size=2,
                           save_frequency=10000 if override else save)
    config['training'].update(total_timesteps=total, eval_frequency=1, use_wandb=False,
                             checkpoint_dir=str(tmp_path / 'checkpoints'),
                             model_save_dir=str(tmp_path / 'models'))
    if override:
        config['training']['save_frequency'] = save
    config['logging']['log_dir'] = str(tmp_path / 'logs')
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    return path, config


def run_cli(module, arguments):
    """Execute the real public command without network or experiment tracking."""
    return subprocess.run([sys.executable, '-m', module, *arguments], cwd=ROOT,
                          env={**os.environ, 'MPLBACKEND': 'Agg',
                               'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize('total,horizon,save,expected,override', [
    (8, 23400, 2, [3, 6, 8], False),
    (2, 23400, 1, [2], True),
    (7, 2, 3, [4, 6], False),
    (3, 23400, 10000, [], False),
])
def test_public_training_counts_actual_steps_and_saves_loadable_models(
        tmp_path, total, horizon, save, expected, override):
    """Check short data, partial final episodes, interval crossings, and fallback."""
    path, config = write_config(tmp_path, total=total, horizon=horizon,
                                save=save, override=override)
    process = run_cli('train.train_td3', ['--config', str(path)])
    assert process.returncode == 0, process.stdout + process.stderr
    assert f'Training completed after {total} action steps' in process.stderr
    assert sorted(p.name for p in (tmp_path / 'checkpoints').glob('checkpoint_steps_*.pth')) == [
        f'checkpoint_steps_{step}.pth' for step in expected
    ]
    final = tmp_path / 'models' / 'final_model.pth'
    checkpoint = torch.load(final, map_location='cpu', weights_only=True)
    assert checkpoint['total_iterations'] == total - 1
    for key in ('actor_state_dict', 'critic_1_state_dict', 'critic_2_state_dict'):
        assert all(torch.isfinite(value).all() for value in checkpoint[key].values())
    for checkpoint_path in (tmp_path / 'checkpoints').glob('checkpoint_steps_*.pth'):
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        steps = int(checkpoint_path.stem.split('_')[-1])
        assert saved['total_iterations'] == steps - 1
    env = create_environment(config)
    restored = load_agent(env, config, str(final))
    assert np.isfinite(restored.select_action(env.reset(), add_noise=False)).all()


def test_public_evaluation_loads_training_checkpoint(tmp_path):
    """Verify train/checkpoint/load/evaluation through both public module CLIs."""
    path, _ = write_config(tmp_path, total=4, save=2)
    trained = run_cli('train.train_td3', ['--config', str(path)])
    assert trained.returncode == 0, trained.stdout + trained.stderr
    evaluated = run_cli('train.evaluate', [
        '--config', str(path), '--model', str(tmp_path / 'models' / 'final_model.pth'),
        '--episodes', '1', '--save_dir', str(tmp_path / 'evaluation'),
    ])
    assert evaluated.returncode == 0, evaluated.stdout + evaluated.stderr
    assert 'Loaded trained agent' in evaluated.stderr
    assert (tmp_path / 'evaluation' / 'evaluation_stats.txt').exists()


@pytest.mark.parametrize('limit,terminal', [(2, False), (3, True)])
def test_budget_truncation_preserves_environment_terminal_flag(tmp_path, limit, terminal):
    """A partial budget bootstraps while genuine CSV exhaustion ends a transition."""
    _, config = write_config(tmp_path)
    env = create_environment(config)
    agent = create_agent(env, config)
    result = train_episode(env, agent, MetricsTracker(), 0,
                           logging.getLogger('budget-test'), step_limit=limit)
    assert result['episode_steps'] == limit
    assert result['budget_truncated'] is not terminal
    assert bool(agent.replay_buffer.dones[limit - 1]) is terminal


@pytest.mark.parametrize('field', ['total_timesteps', 'eval_frequency', 'save_frequency', 'max_episode_steps'])
@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_schedule_rejects_nonpositive_or_noninteger_steps(tmp_path, field, value):
    """Reject invalid intervals rather than rounding them into misleading schedules."""
    _, config = write_config(tmp_path)
    section = 'environment' if field == 'max_episode_steps' else 'training'
    config[section][field] = value
    with pytest.raises(ValueError, match=field + ' must be a positive integer'):
        training_schedule(config)


def test_invalid_schedule_fails_before_creating_training_outputs(tmp_path):
    """The public CLI rejects zero intervals before models, logs, or tracking."""
    path, config = write_config(tmp_path, save=0)
    process = run_cli('train.train_td3', ['--config', str(path)])
    assert process.returncode == 2
    assert 'save_frequency must be a positive integer' in process.stderr
    assert not Path(config['logging']['log_dir']).exists()
    assert not Path(config['training']['model_save_dir']).exists()
