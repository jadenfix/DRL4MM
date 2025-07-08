"""
Tests for TD3 Agent
"""

import pytest
import numpy as np
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.td3_agent import TD3Agent
from agent.networks import Actor, Critic
from agent.replay_buffer import ReplayBuffer


class TestTD3Agent:
    """Test cases for TD3Agent class"""
    
    def test_initialization(self):
        """Test agent initialization"""
        agent = TD3Agent(
            state_dim=50,
            action_dim=2,
            max_action=1.0
        )
        
        assert agent.state_dim == 50
        assert agent.action_dim == 2
        assert agent.max_action == 1.0
        assert agent.training == True
        assert agent.total_iterations == 0
        assert isinstance(agent.actor, Actor)
        assert isinstance(agent.critic_1, Critic)
        assert isinstance(agent.critic_2, Critic)
        assert isinstance(agent.replay_buffer, ReplayBuffer)
    
    def test_action_selection(self):
        """Test action selection"""
        agent = TD3Agent(
            state_dim=50,
            action_dim=2,
            max_action=1.0
        )
        
        state = np.random.randn(50)
        
        # Test with noise
        action = agent.select_action(state, add_noise=True)
        assert action.shape == (2,)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)
        
        # Test without noise
        action = agent.select_action(state, add_noise=False)
        assert action.shape == (2,)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)
    
    def test_store_transition(self):
        """Test storing transitions"""
        agent = TD3Agent(
            state_dim=50,
            action_dim=2,
            max_action=1.0
        )
        
        state = np.random.randn(50)
        action = np.random.randn(2)
        reward = 1.0
        next_state = np.random.randn(50)
        done = False
        
        initial_size = len(agent.replay_buffer)
        agent.store_transition(state, action, reward, next_state, done)
        
        assert len(agent.replay_buffer) == initial_size + 1
    
    def test_train_insufficient_data(self):
        """Test training with insufficient data"""
        agent = TD3Agent(
            state_dim=50,
            action_dim=2,
            max_action=1.0
        )
        
        # Try to train with empty buffer
        metrics = agent.train()
        assert metrics == {}
    
    def test_train_with_data(self):
        """Test training with sufficient data"""
        agent = TD3Agent(
            state_dim=50,
            action_dim=2,
            max_action=1.0
        )
        
        # Fill buffer with minimum required data
        batch_size = agent.batch_size
        for _ in range(batch_size):
            state = np.random.randn(50)
            action = np.random.randn(2)
            reward = np.random.randn()
            next_state = np.random.randn(50)
            done = np.random.choice([True, False])
            
            agent.store_transition(state, action, reward, next_state, done)
        
        # Train
        metrics = agent.train()
        
        assert 'critic_loss' in metrics
        assert 'total_iterations' in metrics
        assert 'buffer_size' in metrics
        assert 'exploration_noise' in metrics
        assert agent.total_iterations == 1
    
    def test_eval_mode(self):
        """Test evaluation mode"""
        agent = TD3Agent(
            state_dim=50,
            action_dim=2,
            max_action=1.0
        )
        
        # Set to eval mode
        agent.set_eval_mode()
        assert agent.training == False
        
        # Set back to train mode
        agent.set_train_mode()
        assert agent.training == True
    
    def test_get_stats(self):
        """Test getting agent statistics"""
        agent = TD3Agent(
            state_dim=50,
            action_dim=2,
            max_action=1.0
        )
        
        stats = agent.get_stats()
        
        assert 'total_iterations' in stats
        assert 'buffer_size' in stats
        assert 'exploration_noise' in stats
        assert 'avg_episode_reward' in stats
        assert 'avg_actor_loss' in stats
        assert 'avg_critic_loss' in stats
        assert 'training_mode' in stats
    
    def test_add_episode_reward(self):
        """Test adding episode rewards"""
        agent = TD3Agent(
            state_dim=50,
            action_dim=2,
            max_action=1.0
        )
        
        # Add some rewards
        agent.add_episode_reward(100.0)
        agent.add_episode_reward(200.0)
        agent.add_episode_reward(150.0)
        
        stats = agent.get_stats()
        assert stats['avg_episode_reward'] == 150.0


class TestReplayBuffer:
    """Test cases for ReplayBuffer class"""
    
    def test_initialization(self):
        """Test replay buffer initialization"""
        buffer = ReplayBuffer(
            state_dim=50,
            action_dim=2,
            max_size=1000
        )
        
        assert buffer.max_size == 1000
        assert buffer.size == 0
        assert buffer.ptr == 0
        assert len(buffer) == 0
    
    def test_add_transitions(self):
        """Test adding transitions to buffer"""
        buffer = ReplayBuffer(
            state_dim=50,
            action_dim=2,
            max_size=1000
        )
        
        # Add transition
        state = np.random.randn(50)
        action = np.random.randn(2)
        reward = 1.0
        next_state = np.random.randn(50)
        done = False
        
        buffer.add(state, action, reward, next_state, done)
        
        assert len(buffer) == 1
        assert buffer.ptr == 1
    
    def test_sample_transitions(self):
        """Test sampling transitions from buffer"""
        buffer = ReplayBuffer(
            state_dim=50,
            action_dim=2,
            max_size=1000
        )
        
        # Add multiple transitions
        for _ in range(100):
            state = np.random.randn(50)
            action = np.random.randn(2)
            reward = np.random.randn()
            next_state = np.random.randn(50)
            done = np.random.choice([True, False])
            
            buffer.add(state, action, reward, next_state, done)
        
        # Sample batch
        batch = buffer.sample(32)
        states, actions, rewards, next_states, dones = batch
        
        assert states.shape == (32, 50)
        assert actions.shape == (32, 2)
        assert rewards.shape == (32,)
        assert next_states.shape == (32, 50)
        assert dones.shape == (32,)
    
    def test_buffer_overflow(self):
        """Test buffer overflow behavior"""
        buffer = ReplayBuffer(
            state_dim=50,
            action_dim=2,
            max_size=10
        )
        
        # Add more transitions than buffer size
        for i in range(15):
            state = np.random.randn(50)
            action = np.random.randn(2)
            reward = float(i)
            next_state = np.random.randn(50)
            done = False
            
            buffer.add(state, action, reward, next_state, done)
        
        # Buffer should be full
        assert len(buffer) == 10
        assert buffer.ptr == 5  # Should wrap around
    
    def test_clear_buffer(self):
        """Test clearing the buffer"""
        buffer = ReplayBuffer(
            state_dim=50,
            action_dim=2,
            max_size=1000
        )
        
        # Add transitions
        for _ in range(10):
            state = np.random.randn(50)
            action = np.random.randn(2)
            reward = np.random.randn()
            next_state = np.random.randn(50)
            done = False
            
            buffer.add(state, action, reward, next_state, done)
        
        # Clear buffer
        buffer.clear()
        
        assert len(buffer) == 0
        assert buffer.ptr == 0


class TestNetworks:
    """Test cases for neural networks"""
    
    def test_actor_network(self):
        """Test Actor network"""
        config = {'hidden_dims': [64, 32], 'activation': 'relu'}
        
        actor = Actor(
            state_dim=50,
            action_dim=2,
            max_action=1.0,
            config=config
        )
        
        # Test forward pass
        state = torch.randn(32, 50)
        action = actor(state)
        
        assert action.shape == (32, 2)
        assert torch.all(action >= -1.0)
        assert torch.all(action <= 1.0)
    
    def test_critic_network(self):
        """Test Critic network"""
        config = {'hidden_dims': [64, 32], 'activation': 'relu'}
        
        critic = Critic(
            state_dim=50,
            action_dim=2,
            config=config
        )
        
        # Test forward pass
        state = torch.randn(32, 50)
        action = torch.randn(32, 2)
        q_value = critic(state, action)
        
        assert q_value.shape == (32, 1)


if __name__ == "__main__":
    pytest.main([__file__])
