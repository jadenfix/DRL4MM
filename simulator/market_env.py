"""
Market Making RL Environment

This module provides a Gym-compatible environment for training market making agents
using real limit order book data and realistic trading simulation.
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from collections import deque
import os
from dotenv import load_dotenv

from .lob_simulator import LOBSimulator, OrderSide, Trade
from .historical import load_events

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """
    Represents the current market state for the agent
    """
    timestamp: float
    mid_price: float
    spread: float
    imbalance: float
    inventory: int
    unrealized_pnl: float
    realized_pnl: float
    position_value: float
    quotes_outstanding: int
    recent_fills: List[Trade]
    lob_features: np.ndarray
    flow_features: np.ndarray


class MarketMakingEnv(gym.Env):
    """
    Market making environment for reinforcement learning
    
    The agent observes limit order book state and recent market activity,
    then decides on bid/ask quote offsets to maximize PnL while managing inventory.
    """
    
    def __init__(self, 
                 symbol: str = "AAPL",
                 config: Dict = None,
                 data_source: str = "synthetic"):
        """
        Initialize the market making environment
        
        Args:
            symbol: Trading symbol
            config: Configuration dictionary
            data_source: Data source type ('synthetic', 'historical', 'live')
        """
        super().__init__()
        
        # Load environment variables
        load_dotenv('secrets.env')
        
        self.symbol = symbol
        self.config = {**self._default_config(), **(config or {})}
        self.data_source = data_source
        
        # Initialize simulator
        self.simulator = LOBSimulator(
            symbol=symbol,
            tick_size=self.config['tick_size'],
            latency_mean=self.config['latency_mean'],
            latency_std=self.config['latency_std']
        )
        
        # Environment parameters
        self.max_inventory = self.config['max_inventory']
        self.inventory_penalty = self.config['inventory_penalty']
        self.tick_size = self.config['tick_size']
        self.min_spread = self.config['min_spread']
        self.max_spread = self.config['max_spread']
        self.quote_ttl = self.config['quote_ttl']
        
        # State tracking
        self.current_time = 0.0
        self.episode_start_time = 0.0
        self.inventory = 0
        self.cash = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.position_value = 0.0
        self._entry_price = 0.0
        self._processed_trade_count = 0
        self._rewarded_fills = 0
        self._episode_steps = 0
        self._historical_cursor = 0
        self._history_exhausted = False
        
        # Order management
        self.active_quotes = {}  # {'bid': order_id, 'ask': order_id}
        self.recent_fills = deque(maxlen=100)
        
        # Feature processing
        self.lob_depth = self.config['lob_depth']
        self.feature_window = self.config['feature_window']
        self.price_history = deque(maxlen=self.feature_window)
        self.volume_history = deque(maxlen=self.feature_window)
        self.trade_history = deque(maxlen=self.feature_window)
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Performance tracking
        self.episode_trades = 0
        self.episode_volume = 0
        self.episode_pnl = 0.0
        self.fill_count = 0
        
        # Market data (will be populated based on data_source)
        self.market_data = None
        self.data_iterator = None
        
        logger.info(f"Initialized MarketMakingEnv for {symbol}")
    
    def _default_config(self) -> Dict:
        """
        Default configuration parameters
        """
        return {
            'max_inventory': 1000,
            'inventory_penalty': 0.001,
            'tick_size': 0.01,
            'min_spread': 0.01,
            'max_spread': 1.0,
            'quote_ttl': 10.0,  # seconds
            'latency_mean': 0.001,
            'latency_std': 0.0005,
            'lob_depth': 10,
            'feature_window': 100,
            'max_episode_steps': 23400,  # 6.5 hours
            'reward_scaling': 1.0,
            'fill_bonus': 0.1,
            'spread_penalty': 0.0001
        }
    
    def _setup_spaces(self):
        """
        Setup action and observation spaces
        """
        # Action space: [bid_offset, ask_offset] as percentage of mid price
        # Values between -0.1 and 0.1 (±10% of mid price)
        self.action_space = spaces.Box(
            low=np.array([-0.1, -0.1]),
            high=np.array([0.1, 0.1]),
            dtype=np.float32
        )
        
        # Observation space components:
        # 1. LOB features: bid/ask prices and volumes (lob_depth * 4)
        # 2. Market features: mid_price, spread, imbalance, volatility (4)
        # 3. Agent state: inventory, unrealized_pnl, position_value, quotes_outstanding (4)
        # 4. Time features: time_of_day, time_since_last_fill (2)
        # 5. Flow features: recent trade imbalance, volume flow (2)
        
        obs_dim = (
            self.lob_depth * 4 +  # LOB features
            4 +  # Market features
            4 +  # Agent state
            2 +  # Time features
            2    # Flow features
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        logger.info(f"Action space: {self.action_space}")
        logger.info(f"Observation space: {self.observation_space}")
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset environment to initial state
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset simulator
        self.simulator.reset()
        
        # Reset state
        self.current_time = 0.0
        self.episode_start_time = 0.0
        self.inventory = 0
        self.cash = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.position_value = 0.0
        self._entry_price = 0.0
        self._processed_trade_count = 0
        self._rewarded_fills = 0
        self._episode_steps = 0
        self._historical_cursor = 0
        self._history_exhausted = False
        
        # Reset tracking
        self.active_quotes = {}
        self.recent_fills.clear()
        self.episode_trades = 0
        self.episode_volume = 0
        self.episode_pnl = 0.0
        self.fill_count = 0
        
        # Reset feature histories
        self.price_history.clear()
        self.volume_history.clear()
        self.trade_history.clear()
        
        # Initialize market data
        self._initialize_market_data()
        
        # Get initial observation
        obs = self._get_observation()
        
        logger.info(f"Environment reset for episode")
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step within the environment
        
        Args:
            action: [bid_offset, ask_offset] as percentage of mid price
            
        Returns:
            observation: New state
            reward: Reward for this step
            done: Whether episode has ended
            info: Additional information
        """
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Action must contain two finite bid/ask offsets")
        if not self.action_space.contains(action):
            raise ValueError("Action offsets exceed the environment's action space")

        if self.data_source == 'historical':
            if self._history_exhausted or self._is_episode_done():
                raise RuntimeError('Historical episode ended; call reset before stepping again')
            # Quotes act on the current observation before the next external event.
            self._cancel_existing_quotes()
            self._place_quotes(action)
            self._advance_historical_event()
        else:
            self.current_time += 1.0
        self._episode_steps += 1
        
        # Update simulator
        sim_state = self.simulator.step(self.current_time)
        
        # Process market data update
        self._process_market_update(sim_state)
        
        if self.data_source != 'historical':
            self._cancel_existing_quotes()
            self._place_quotes(action)
        
        # Update agent state
        self._update_agent_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Get new observation
        obs = self._get_observation()
        
        # Prepare info dictionary
        info = self._get_info()
        
        return obs, reward, done, info
    
    def _initialize_market_data(self):
        """
        Initialize market data source
        """
        if self.data_source == "synthetic":
            # Initialize synthetic market data generator
            self._initialize_synthetic_data()
        elif self.data_source == "historical":
            # Load historical data
            self._load_historical_data()
        elif self.data_source == "live":
            # Connect to live data feed
            self._connect_live_data()
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")
    
    def _initialize_synthetic_data(self):
        """
        Initialize synthetic market data generator
        """
        # Create synthetic LOB with reasonable parameters
        base_price = 150.0
        
        # Add some initial depth to both sides
        for i in range(10):
            # Bid side
            bid_price = base_price - (i + 1) * self.tick_size
            bid_qty = np.random.randint(100, 1000)
            # Ask side
            ask_price = base_price + (i + 1) * self.tick_size
            ask_qty = np.random.randint(100, 1000)
    
    def _load_historical_data(self):
        """
        Load historical market data from files
        """
        path = self.config.get('historical_data_path')
        if not path:
            raise ValueError('historical_data_path is required for historical replay')
        self.market_data = load_events(path)
        # Prime only through the first two-sided book, leaving future rows unseen.
        while not (self.simulator.lob.bids and self.simulator.lob.asks):
            if self._historical_cursor >= len(self.market_data):
                raise ValueError('Replay never establishes an initial two-sided book')
            self._advance_historical_event()
        if self._history_exhausted:
            raise ValueError('Replay needs events after its initial two-sided book')
        self.episode_start_time = self.current_time
        self._processed_trade_count = self.simulator.lob.trade_count
        self._process_market_update(self.simulator.step(self.current_time))

    def _advance_historical_event(self):
        """Expire existing orders, apply the next event, then advance the cursor."""
        event = self.market_data[self._historical_cursor]
        self.current_time = event.timestamp
        self.simulator.step(self.current_time)
        event.apply(self.simulator.lob)
        self._historical_cursor += 1
        self._history_exhausted = self._historical_cursor == len(self.market_data)
    
    def _connect_live_data(self):
        """
        Connect to live market data feed
        """
        raise NotImplementedError('Live exchange data is not implemented; use historical CSV replay')
    
    def _process_market_update(self, sim_state: Dict):
        """
        Process market data update
        """
        # Update price history
        if sim_state['mid_price'] is not None:
            self.price_history.append(sim_state['mid_price'])
        
        # Update volume history
        total_volume = sum(
            level['quantity'] for level in sim_state['depth']['bids'] + sim_state['depth']['asks']
        )
        self.volume_history.append(total_volume)
        
    
    def _cancel_existing_quotes(self):
        """
        Cancel any existing quotes
        """
        for side, order_id in self.active_quotes.items():
            if order_id:
                self.simulator.cancel_agent_order(order_id)
        
        self.active_quotes = {}
    
    def _place_quotes(self, action: np.ndarray):
        """
        Place new quotes based on action
        """
        mid_price = self.simulator.lob.get_mid_price()
        if mid_price is None:
            return
        
        bid_offset, ask_offset = action
        
        # Calculate quote prices
        bid_price = mid_price + (bid_offset * mid_price)
        ask_price = mid_price + (ask_offset * mid_price)
        
        # Ensure minimum spread
        if ask_price - bid_price < self.min_spread:
            spread_adjustment = (self.min_spread - (ask_price - bid_price)) / 2
            bid_price -= spread_adjustment
            ask_price += spread_adjustment
        
        # Round to tick size
        bid_price = round(bid_price / self.tick_size) * self.tick_size
        ask_price = round(ask_price / self.tick_size) * self.tick_size
        
        # Place orders (fixed quantity for now)
        quote_size = self.config.get('quote_size', 100)
        if isinstance(quote_size, bool) or not isinstance(quote_size, int) or quote_size <= 0:
            raise ValueError('quote_size must be a positive integer')
        
        # Place bid
        if bid_price > 0:
            bid_order_id = self.simulator.place_agent_order(
                side="buy",
                price=bid_price,
                quantity=quote_size,
                ttl=self.quote_ttl
            )
            self.active_quotes['bid'] = bid_order_id
        
        # Place ask
        if ask_price > 0:
            ask_order_id = self.simulator.place_agent_order(
                side="sell",
                price=ask_price,
                quantity=quote_size,
                ttl=self.quote_ttl
            )
            self.active_quotes['ask'] = ask_order_id
    
    def _update_agent_state(self):
        """
        Consume each newly executed trade once and mark the open position.

        Lifetime accounting must not depend on the bounded display history.
        Realized PnL uses average entry cost; cash plus marked position equals
        realized plus unrealized PnL (the simulator currently charges no fees).
        """
        book = self.simulator.lob
        pending = book.trade_count - self._processed_trade_count
        if pending < 0 or pending > len(book.trades):
            raise RuntimeError("Trade history gap: cannot reconcile agent accounting")
        for fill in list(book.trades)[len(book.trades) - pending:]:
            bought = fill.buyer_id.startswith("agent_")
            sold = fill.seller_id.startswith("agent_")
            # Neither third-party trades nor self trades earn economic rewards.
            if bought == sold:
                continue
            quantity = fill.quantity if bought else -fill.quantity
            previous = self.inventory
            if previous == 0 or previous * quantity > 0:
                self._entry_price = (
                    abs(previous) * self._entry_price + abs(quantity) * fill.price
                ) / (abs(previous) + abs(quantity))
            else:
                closed = min(abs(previous), abs(quantity))
                self.realized_pnl += closed * (fill.price - self._entry_price) * np.sign(previous)
                if abs(quantity) > abs(previous):
                    self._entry_price = fill.price
                elif abs(quantity) == abs(previous):
                    self._entry_price = 0.0
            self.inventory += quantity
            self.cash -= quantity * fill.price
            self.recent_fills.append(fill)
            self.trade_history.append(quantity)
            self.fill_count += 1
            self.episode_trades += 1
            self.episode_volume += fill.quantity
        self._processed_trade_count = book.trade_count
        mid_price = self.simulator.lob.get_mid_price()
        if mid_price is not None:
            self.position_value = self.inventory * mid_price
            self.unrealized_pnl = self.inventory * (mid_price - self._entry_price)
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward for current step
        """
        # Basic reward: change in PnL
        delta_pnl = self.realized_pnl + self.unrealized_pnl - self.episode_pnl
        self.episode_pnl = self.realized_pnl + self.unrealized_pnl
        
        # Inventory penalty
        inventory_penalty = self.inventory_penalty * abs(self.inventory)
        
        # Fill bonus
        fill_bonus = 0.0
        new_fills = self.fill_count - self._rewarded_fills
        if new_fills > 0:
            fill_bonus = self.config['fill_bonus'] * new_fills
            self._rewarded_fills = self.fill_count
        
        # Spread penalty (encourage tighter spreads)
        spread = self.simulator.lob.get_spread()
        spread_penalty = 0.0
        if spread is not None:
            spread_penalty = self.config['spread_penalty'] * spread
        
        # Total reward
        reward = delta_pnl - inventory_penalty + fill_bonus - spread_penalty
        
        return reward * self.config['reward_scaling']
    
    def _is_episode_done(self) -> bool:
        """
        Check if episode should end
        """
        # End if max steps reached
        if self._episode_steps >= self.config['max_episode_steps']:
            return True

        if self.data_source == 'historical' and self._history_exhausted:
            return True
        
        # End if inventory is too large
        if abs(self.inventory) >= self.max_inventory:
            return True
        
        # End if significant loss
        if self.realized_pnl + self.unrealized_pnl < -10000:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation vector
        """
        features = []
        
        # 1. LOB features
        depth = self.simulator.lob.get_market_depth(self.lob_depth)
        
        # Bid features
        for i in range(self.lob_depth):
            if i < len(depth['bids']):
                level = depth['bids'][i]
                features.extend([level['price'], level['quantity']])
            else:
                features.extend([0.0, 0.0])
        
        # Ask features
        for i in range(self.lob_depth):
            if i < len(depth['asks']):
                level = depth['asks'][i]
                features.extend([level['price'], level['quantity']])
            else:
                features.extend([0.0, 0.0])
        
        # 2. Market features
        mid_price = self.simulator.lob.get_mid_price() or 0.0
        spread = self.simulator.lob.get_spread() or 0.0
        imbalance = self.simulator.lob.get_order_book_imbalance()
        volatility = self._calculate_volatility()
        
        features.extend([mid_price, spread, imbalance, volatility])
        
        # 3. Agent state
        features.extend([
            self.inventory,
            self.unrealized_pnl,
            self.position_value,
            self._active_quote_count()
        ])
        
        # 4. Time features
        time_of_day = (self.current_time % 23400) / 23400  # Normalized to [0, 1]
        time_since_last_fill = self._time_since_last_fill()
        features.extend([time_of_day, time_since_last_fill])
        
        # 5. Flow features
        trade_imbalance = self._calculate_trade_imbalance()
        volume_flow = self._calculate_volume_flow()
        features.extend([trade_imbalance, volume_flow])
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_volatility(self) -> float:
        """
        Calculate recent price volatility
        """
        if len(self.price_history) < 2:
            return 0.0
        
        prices = np.array(self.price_history)
        returns = np.diff(np.log(prices))
        return np.std(returns)
    
    def _time_since_last_fill(self) -> float:
        """
        Calculate time since last fill
        """
        if not self.recent_fills:
            return 1.0  # Normalized max value
        
        last_fill_time = self.recent_fills[-1].timestamp
        time_diff = self.current_time - last_fill_time
        return min(time_diff / 60.0, 1.0)  # Normalize to [0, 1], max 60 seconds
    
    def _calculate_trade_imbalance(self) -> float:
        """
        Calculate recent trade flow imbalance
        """
        if not self.trade_history:
            return 0.0
        
        # Simple implementation - can be enhanced
        recent_trades = list(self.trade_history)[-10:]  # Last 10 trades
        buy_volume = sum(t for t in recent_trades if t > 0)
        sell_volume = sum(abs(t) for t in recent_trades if t < 0)
        
        if buy_volume + sell_volume == 0:
            return 0.0
        
        return (buy_volume - sell_volume) / (buy_volume + sell_volume)
    
    def _calculate_volume_flow(self) -> float:
        """
        Calculate volume flow indicator
        """
        if len(self.volume_history) < 2:
            return 0.0
        
        recent_volume = self.volume_history[-1]
        avg_volume = np.mean(self.volume_history)
        
        if avg_volume == 0:
            return 0.0
        
        return (recent_volume - avg_volume) / avg_volume
    
    def _get_info(self) -> Dict:
        """
        Get additional information for this step
        """
        return {
            'inventory': self.inventory,
            'cash': self.cash,
            'equity': self.cash + self.position_value,
            'timestamp': self.current_time,
            'episode_steps': self._episode_steps,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'position_value': self.position_value,
            'fill_count': self.fill_count,
            'episode_trades': self.episode_trades,
            'episode_volume': self.episode_volume,
            'mid_price': self.simulator.lob.get_mid_price(),
            'spread': self.simulator.lob.get_spread(),
            'imbalance': self.simulator.lob.get_order_book_imbalance(),
            'active_quotes': self._active_quote_count()
        }

    def _active_quote_count(self):
        """Count only agent quotes still resting in the authoritative book."""
        return sum(order_id in self.simulator.lob.orders for order_id in self.active_quotes.values())
    
    def render(self, mode='human'):
        """
        Render the environment
        """
        if mode == 'human':
            print(f"\n=== Market State at t={self.current_time:.1f} ===")
            print(f"Symbol: {self.symbol}")
            print(f"Mid Price: {self.simulator.lob.get_mid_price():.2f}")
            print(f"Spread: {self.simulator.lob.get_spread():.2f}")
            print(f"Inventory: {self.inventory}")
            print(f"Realized PnL: {self.realized_pnl:.2f}")
            print(f"Unrealized PnL: {self.unrealized_pnl:.2f}")
            print(f"Active Quotes: {len(self.active_quotes)}")
            print(f"Recent Fills: {len(self.recent_fills)}")
    
    def close(self):
        """
        Clean up environment
        """
        pass

