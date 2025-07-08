"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) Agent for Market Making

This module implements the TD3 algorithm specifically designed for market making
with continuous action spaces for bid/ask quote placement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque
import random

from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class TD3Agent:
    """
    TD3 Agent for Market Making
    
    Twin Delayed Deep Deterministic Policy Gradient with improvements:
    - Clipped Double Q-Learning
    - Delayed Policy Updates
    - Target Policy Smoothing
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 max_action: float = 1.0,
                 config: Dict = None):
        """
        Initialize TD3 agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space  
            max_action: Maximum action value
            config: Configuration dictionary
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.config = config or self._default_config()
        
        # Set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config['use_cuda'] else "cpu"
        )
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, max_action, self.config).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action, self.config).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic_1 = Critic(state_dim, action_dim, self.config).to(self.device)
        self.critic_1_target = Critic(state_dim, action_dim, self.config).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        
        self.critic_2 = Critic(state_dim, action_dim, self.config).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim, self.config).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config['actor_lr'])
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.config['critic_lr'])
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.config['critic_lr'])
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=self.config['buffer_size'],
            device=self.device
        )
        
        # Training parameters
        self.batch_size = self.config['batch_size']
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']
        self.policy_noise = self.config['policy_noise']
        self.noise_clip = self.config['noise_clip']
        self.policy_freq = self.config['policy_freq']
        
        # Training state
        self.total_iterations = 0
        self.training = True
        
        # Exploration noise
        self.exploration_noise = self.config['exploration_noise']
        self.noise_decay = self.config['noise_decay']
        self.min_noise = self.config['min_noise']
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.actor_losses = deque(maxlen=1000)
        self.critic_losses = deque(maxlen=1000)
        
        logger.info(f"Initialized TD3 Agent with state_dim={state_dim}, action_dim={action_dim}")
    
    def _default_config(self) -> Dict:
        """
        Default configuration for TD3 agent
        """
        return {
            'actor_lr': 3e-4,
            'critic_lr': 3e-4,
            'buffer_size': 1000000,
            'batch_size': 256,
            'gamma': 0.99,
            'tau': 0.005,
            'policy_noise': 0.2,
            'noise_clip': 0.5,
            'policy_freq': 2,
            'exploration_noise': 0.1,
            'noise_decay': 0.995,
            'min_noise': 0.01,
            'use_cuda': True,
            'hidden_dims': [256, 256],
            'activation': 'relu'
        }
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select action using current policy
        
        Args:
            state: Current state
            add_noise: Whether to add exploration noise
            
        Returns:
            Selected action
        """
        state = torch.FloatTensor(state).to(self.device)
        
        # Get action from actor network
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()
        
        # Add exploration noise if training
        if add_noise and self.training:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        """
        Store transition in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train(self) -> Dict:
        """
        Train the agent using stored transitions
        
        Returns:
            Training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        state, action, reward, next_state, done = batch
        
        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.BoolTensor(done).to(self.device)
        
        # Update critics
        critic_loss = self._update_critics(state, action, reward, next_state, done)
        
        # Update actor and target networks (delayed)
        actor_loss = None
        if self.total_iterations % self.policy_freq == 0:
            actor_loss = self._update_actor(state)
            self._update_target_networks()
        
        self.total_iterations += 1
        
        # Decay exploration noise
        if self.exploration_noise > self.min_noise:
            self.exploration_noise *= self.noise_decay
        
        # Return metrics
        metrics = {
            'critic_loss': critic_loss,
            'total_iterations': self.total_iterations,
            'buffer_size': len(self.replay_buffer),
            'exploration_noise': self.exploration_noise
        }
        
        if actor_loss is not None:
            metrics['actor_loss'] = actor_loss
        
        return metrics
    
    def _update_critics(self, state: torch.Tensor, action: torch.Tensor,
                       reward: torch.Tensor, next_state: torch.Tensor, 
                       done: torch.Tensor) -> float:
        """
        Update critic networks using TD3's clipped double Q-learning
        """
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # Compute target Q-values (take minimum to address overestimation)
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            
            # Compute target
            target = reward + self.gamma * target_q * (~done)
        
        # Current Q-values
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        
        # Compute critic losses
        critic_1_loss = F.mse_loss(current_q1, target)
        critic_2_loss = F.mse_loss(current_q2, target)
        
        # Update critic 1
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        
        # Update critic 2
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        
        # Track losses
        total_critic_loss = (critic_1_loss + critic_2_loss).item()
        self.critic_losses.append(total_critic_loss)
        
        return total_critic_loss
    
    def _update_actor(self, state: torch.Tensor) -> float:
        """
        Update actor network using policy gradient
        """
        # Compute actor loss
        actor_loss = -self.critic_1(state, self.actor(state)).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Track loss
        actor_loss_item = actor_loss.item()
        self.actor_losses.append(actor_loss_item)
        
        return actor_loss_item
    
    def _update_target_networks(self):
        """
        Soft update target networks
        """
        # Update actor target
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Update critic 1 target
        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Update critic 2 target
        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath: str):
        """
        Save agent state
        
        Args:
            filepath: Path to save file
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_1_state_dict': self.critic_1.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_1_optimizer_state_dict': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer_state_dict': self.critic_2_optimizer.state_dict(),
            'total_iterations': self.total_iterations,
            'exploration_noise': self.exploration_noise,
            'config': self.config
        }, filepath)
        
        logger.info(f"Saved agent to {filepath}")
    
    def load(self, filepath: str):
        """
        Load agent state
        
        Args:
            filepath: Path to load file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network states
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        
        # Load optimizer states
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer_state_dict'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer_state_dict'])
        
        # Load training state
        self.total_iterations = checkpoint['total_iterations']
        self.exploration_noise = checkpoint['exploration_noise']
        
        # Update target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        logger.info(f"Loaded agent from {filepath}")
    
    def set_eval_mode(self):
        """
        Set agent to evaluation mode (no exploration)
        """
        self.training = False
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
    
    def set_train_mode(self):
        """
        Set agent to training mode
        """
        self.training = True
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
    
    def get_stats(self) -> Dict:
        """
        Get agent statistics
        
        Returns:
            Dictionary of agent statistics
        """
        return {
            'total_iterations': self.total_iterations,
            'buffer_size': len(self.replay_buffer),
            'exploration_noise': self.exploration_noise,
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'avg_actor_loss': np.mean(self.actor_losses) if self.actor_losses else 0,
            'avg_critic_loss': np.mean(self.critic_losses) if self.critic_losses else 0,
            'training_mode': self.training
        }
    
    def add_episode_reward(self, reward: float):
        """
        Add episode reward for tracking
        
        Args:
            reward: Episode reward
        """
        self.episode_rewards.append(reward)
