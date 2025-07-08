"""
Neural Network Architectures for TD3 Agent

This module defines the Actor and Critic network architectures optimized
for market making with attention to market microstructure features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class Actor(nn.Module):
    """
    Actor Network for TD3 Agent
    
    Maps state to continuous actions (bid/ask offsets)
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float, config: Dict):
        """
        Initialize Actor network
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_action: Maximum action value
            config: Configuration dictionary
        """
        super(Actor, self).__init__()
        
        self.max_action = max_action
        self.hidden_dims = config.get('hidden_dims', [256, 256])
        self.activation = config.get('activation', 'relu')
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation())
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self):
        """Get activation function"""
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'elu':
            return nn.ELU()
        else:
            return nn.ReLU()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor
            
        Returns:
            Action tensor scaled by max_action
        """
        return self.max_action * self.network(state)


class Critic(nn.Module):
    """
    Critic Network for TD3 Agent
    
    Estimates Q-value for state-action pairs
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        """
        Initialize Critic network
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
        """
        super(Critic, self).__init__()
        
        self.hidden_dims = config.get('hidden_dims', [256, 256])
        self.activation = config.get('activation', 'relu')
        
        # Build network layers
        layers = []
        input_dim = state_dim + action_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation())
            input_dim = hidden_dim
        
        # Output layer (Q-value)
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self):
        """Get activation function"""
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'elu':
            return nn.ELU()
        else:
            return nn.ReLU()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Q-value tensor
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class MarketMakingActor(nn.Module):
    """
    Specialized Actor Network for Market Making
    
    Incorporates domain knowledge for market making:
    - Separate processing for LOB features and market state
    - Attention mechanism for LOB levels
    - Specialized output for bid/ask spreads
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float, config: Dict):
        """
        Initialize Market Making Actor
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (should be 2 for bid/ask)
            max_action: Maximum action value
            config: Configuration dictionary
        """
        super(MarketMakingActor, self).__init__()
        
        self.max_action = max_action
        self.lob_depth = config.get('lob_depth', 10)
        self.hidden_dims = config.get('hidden_dims', [256, 256])
        
        # LOB feature dimensions (price, volume for each bid/ask level)
        self.lob_features_dim = self.lob_depth * 4  # 2 for bids, 2 for asks
        
        # Other feature dimensions
        self.market_features_dim = 4  # mid_price, spread, imbalance, volatility
        self.agent_features_dim = 4   # inventory, pnl, position_value, quotes
        self.time_features_dim = 2    # time_of_day, time_since_fill
        self.flow_features_dim = 2    # trade_imbalance, volume_flow
        
        # LOB processing network
        self.lob_encoder = nn.Sequential(
            nn.Linear(self.lob_features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Market state processing network
        other_features_dim = (self.market_features_dim + self.agent_features_dim + 
                             self.time_features_dim + self.flow_features_dim)
        self.market_encoder = nn.Sequential(
            nn.Linear(other_features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Combined feature processing
        combined_dim = 64 + 32  # LOB + market features
        self.combined_network = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in [self.lob_encoder, self.market_encoder, self.combined_network]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with specialized market making architecture
        
        Args:
            state: State tensor
            
        Returns:
            Action tensor [bid_offset, ask_offset]
        """
        # Split state into components
        lob_features = state[:, :self.lob_features_dim]
        market_features = state[:, self.lob_features_dim:]
        
        # Process LOB features
        lob_encoded = self.lob_encoder(lob_features)
        
        # Process market features
        market_encoded = self.market_encoder(market_features)
        
        # Combine features
        combined = torch.cat([lob_encoded, market_encoded], dim=1)
        
        # Generate actions
        actions = self.combined_network(combined)
        
        return self.max_action * actions


class MarketMakingCritic(nn.Module):
    """
    Specialized Critic Network for Market Making
    
    Incorporates domain knowledge for value estimation in market making context
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        """
        Initialize Market Making Critic
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
        """
        super(MarketMakingCritic, self).__init__()
        
        self.lob_depth = config.get('lob_depth', 10)
        self.hidden_dims = config.get('hidden_dims', [256, 256])
        
        # Feature dimensions
        self.lob_features_dim = self.lob_depth * 4
        self.other_features_dim = state_dim - self.lob_features_dim
        
        # LOB processing network
        self.lob_encoder = nn.Sequential(
            nn.Linear(self.lob_features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # State-action processing network
        combined_dim = 64 + self.other_features_dim + action_dim
        self.value_network = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in [self.lob_encoder, self.value_network]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Q-value tensor
        """
        # Split state into components
        lob_features = state[:, :self.lob_features_dim]
        other_features = state[:, self.lob_features_dim:]
        
        # Process LOB features
        lob_encoded = self.lob_encoder(lob_features)
        
        # Combine all features
        combined = torch.cat([lob_encoded, other_features, action], dim=1)
        
        # Estimate Q-value
        return self.value_network(combined)


class LSTMActor(nn.Module):
    """
    LSTM-based Actor for sequential market making decisions
    
    Incorporates temporal dependencies in market data
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float, config: Dict):
        """
        Initialize LSTM Actor
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_action: Maximum action value
            config: Configuration dictionary
        """
        super(LSTMActor, self).__init__()
        
        self.max_action = max_action
        self.hidden_size = config.get('lstm_hidden_size', 128)
        self.num_layers = config.get('lstm_num_layers', 2)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2 if self.num_layers > 1 else 0
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.output_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state: torch.Tensor, hidden: tuple = None) -> Tuple[torch.Tensor, tuple]:
        """
        Forward pass
        
        Args:
            state: State tensor (batch_size, seq_len, state_dim)
            hidden: Hidden state tuple (h_0, c_0)
            
        Returns:
            Action tensor and new hidden state
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(state, hidden)
        
        # Take last timestep output
        if lstm_out.dim() == 3:
            lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Generate actions
        actions = self.output_layers(lstm_out)
        
        return self.max_action * actions, hidden
