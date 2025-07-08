"""
Replay Buffer for TD3 Agent

This module implements an efficient replay buffer for storing and sampling
transitions during training with optional prioritized experience replay.
"""

import numpy as np
import torch
from typing import Tuple, Optional
import random
from collections import deque


class ReplayBuffer:
    """
    Replay buffer for TD3 agent with uniform sampling
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 1000000, 
                 device: torch.device = torch.device('cpu')):
        """
        Initialize replay buffer
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_size: Maximum buffer size
            device: Device to store tensors
        """
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Initialize storage
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=bool)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool):
        """
        Add a transition to the buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                             np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return self.size
    
    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    
    Implements prioritized sampling based on TD errors
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 1000000,
                 alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001,
                 device: torch.device = torch.device('cpu')):
        """
        Initialize prioritized replay buffer
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_size: Maximum buffer size
            alpha: Prioritization exponent
            beta: Importance sampling exponent
            beta_increment: Beta increment per sample
            device: Device to store tensors
        """
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Initialize storage
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=bool)
        
        # Priority storage
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool, td_error: Optional[float] = None):
        """
        Add a transition to the buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            td_error: TD error for prioritization
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        # Set priority
        if td_error is not None:
            priority = (abs(td_error) + 1e-6) ** self.alpha
        else:
            priority = self.max_priority
        
        self.priorities[self.ptr] = priority
        self.max_priority = max(self.max_priority, priority)
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions with prioritized sampling
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            weights.astype(np.float32),
            indices
        )
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD errors
        
        Args:
            indices: Indices of transitions to update
            td_errors: New TD errors
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return self.size
    
    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0
        self.max_priority = 1.0


class EpisodeReplayBuffer:
    """
    Episode-based replay buffer for market making
    
    Stores complete episodes and allows for episode-based sampling
    """
    
    def __init__(self, max_episodes: int = 1000):
        """
        Initialize episode replay buffer
        
        Args:
            max_episodes: Maximum number of episodes to store
        """
        self.max_episodes = max_episodes
        self.episodes = deque(maxlen=max_episodes)
        self.episode_rewards = deque(maxlen=max_episodes)
        self.episode_lengths = deque(maxlen=max_episodes)
    
    def add_episode(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                   next_states: np.ndarray, dones: np.ndarray):
        """
        Add a complete episode to the buffer
        
        Args:
            states: Episode states
            actions: Episode actions
            rewards: Episode rewards
            next_states: Episode next states
            dones: Episode done flags
        """
        episode = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
        
        self.episodes.append(episode)
        self.episode_rewards.append(rewards.sum())
        self.episode_lengths.append(len(states))
    
    def sample_episodes(self, num_episodes: int) -> list:
        """
        Sample complete episodes
        
        Args:
            num_episodes: Number of episodes to sample
            
        Returns:
            List of episode dictionaries
        """
        if num_episodes > len(self.episodes):
            num_episodes = len(self.episodes)
        
        return random.sample(list(self.episodes), num_episodes)
    
    def sample_transitions(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray]:
        """
        Sample transitions from stored episodes
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if len(self.episodes) == 0:
            return None
        
        # Collect all transitions
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []
        
        for episode in self.episodes:
            all_states.append(episode['states'])
            all_actions.append(episode['actions'])
            all_rewards.append(episode['rewards'])
            all_next_states.append(episode['next_states'])
            all_dones.append(episode['dones'])
        
        # Concatenate all transitions
        states = np.concatenate(all_states)
        actions = np.concatenate(all_actions)
        rewards = np.concatenate(all_rewards)
        next_states = np.concatenate(all_next_states)
        dones = np.concatenate(all_dones)
        
        # Sample transitions
        indices = np.random.choice(len(states), batch_size, replace=False)
        
        return (
            states[indices],
            actions[indices],
            rewards[indices],
            next_states[indices],
            dones[indices]
        )
    
    def get_episode_stats(self) -> dict:
        """
        Get episode statistics
        
        Returns:
            Dictionary of episode statistics
        """
        if len(self.episode_rewards) == 0:
            return {'mean_reward': 0, 'std_reward': 0, 'mean_length': 0}
        
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'num_episodes': len(self.episodes)
        }
    
    def __len__(self) -> int:
        """Return number of stored episodes"""
        return len(self.episodes)
    
    def clear(self):
        """Clear the buffer"""
        self.episodes.clear()
        self.episode_rewards.clear()
        self.episode_lengths.clear()
