"""Exercise synthetic market observations through real TD3 updates."""
import numpy as np
import torch
from agent.td3_agent import TD3Agent
from simulator.market_env import MarketMakingEnv


def test_synthetic_episode_collects_replay_and_updates_agent():
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        env = MarketMakingEnv(data_source="synthetic")
        state = env.reset(seed=7)
        agent = TD3Agent(state_dim=len(state), action_dim=2)
        agent.batch_size = 8
        for _ in range(16):
            action = agent.select_action(state, add_noise=False)
            next_state, reward, done, _ = env.step(action)
            assert np.isfinite(next_state).all() and np.isfinite(reward)
            agent.store_transition(state, action, reward, next_state, done)
            metrics = agent.train()
            state = next_state
        assert len(agent.replay_buffer) == 16
        assert agent.total_iterations == 9
        assert np.isfinite(metrics["critic_loss"])
    finally:
        torch.set_num_threads(original_threads)
