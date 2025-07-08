"""
Data utilities for market making RL project

This module provides utilities for loading, processing, and generating market data
for training the market making agent.
"""

import numpy as np
import pandas as pd
import requests
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
import time

logger = logging.getLogger(__name__)


class NasdaqDataLoader:
    """
    Loader for Nasdaq market data via API
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Nasdaq data loader
        
        Args:
            api_key: Nasdaq API key (if None, loads from environment)
        """
        load_dotenv('secrets.env')
        self.api_key = api_key or os.getenv('NASDAQ_API')
        
        if not self.api_key:
            raise ValueError("Nasdaq API key not found. Please set NASDAQ_API environment variable.")
        
        self.base_url = "https://data.nasdaq.com/api/v3"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def get_historical_quotes(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical quote data for a symbol
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with quote data
        """
        # This is a placeholder implementation
        # In practice, you would use the actual Nasdaq API endpoints
        
        url = f"{self.base_url}/datasets/WIKI/{symbol}/data.json"
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'api_key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame (this would need to be adapted for actual API response)
            df = pd.DataFrame(data.get('dataset_data', {}).get('data', []))
            df.columns = data.get('dataset_data', {}).get('column_names', [])
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_level2_data(self, symbol: str, timestamp: str) -> Dict:
        """
        Get Level 2 (depth of book) data for a symbol
        
        Args:
            symbol: Stock symbol
            timestamp: Timestamp for data
            
        Returns:
            Dictionary with LOB data
        """
        # Placeholder implementation
        # In practice, this would fetch real-time or historical LOB data
        
        return {
            'symbol': symbol,
            'timestamp': timestamp,
            'bids': [],
            'asks': [],
            'trades': []
        }


class LOBDataProcessor:
    """
    Processor for Limit Order Book data
    """
    
    def __init__(self, tick_size: float = 0.01):
        """
        Initialize LOB data processor
        
        Args:
            tick_size: Minimum price increment
        """
        self.tick_size = tick_size
    
    def process_lob_snapshot(self, lob_data: Dict) -> Dict:
        """
        Process a single LOB snapshot
        
        Args:
            lob_data: Raw LOB data
            
        Returns:
            Processed LOB features
        """
        bids = lob_data.get('bids', [])
        asks = lob_data.get('asks', [])
        
        # Calculate basic features
        features = {
            'timestamp': lob_data.get('timestamp', 0),
            'symbol': lob_data.get('symbol', ''),
            'bid_price': bids[0]['price'] if bids else 0,
            'ask_price': asks[0]['price'] if asks else 0,
            'bid_size': bids[0]['size'] if bids else 0,
            'ask_size': asks[0]['size'] if asks else 0,
            'spread': 0,
            'mid_price': 0,
            'weighted_mid_price': 0,
            'imbalance': 0,
            'depth_imbalance': 0
        }
        
        if bids and asks:
            features['spread'] = features['ask_price'] - features['bid_price']
            features['mid_price'] = (features['bid_price'] + features['ask_price']) / 2
            
            # Weighted mid price
            total_size = features['bid_size'] + features['ask_size']
            if total_size > 0:
                features['weighted_mid_price'] = (
                    features['bid_price'] * features['ask_size'] + 
                    features['ask_price'] * features['bid_size']
                ) / total_size
            
            # Order imbalance
            features['imbalance'] = (features['bid_size'] - features['ask_size']) / total_size
            
            # Depth imbalance (top 5 levels)
            bid_depth = sum(level['size'] for level in bids[:5])
            ask_depth = sum(level['size'] for level in asks[:5])
            total_depth = bid_depth + ask_depth
            
            if total_depth > 0:
                features['depth_imbalance'] = (bid_depth - ask_depth) / total_depth
        
        return features
    
    def calculate_microprice(self, bid_price: float, ask_price: float,
                           bid_size: float, ask_size: float) -> float:
        """
        Calculate microprice
        
        Args:
            bid_price: Best bid price
            ask_price: Best ask price
            bid_size: Best bid size
            ask_size: Best ask size
            
        Returns:
            Microprice
        """
        if bid_size + ask_size == 0:
            return (bid_price + ask_price) / 2
        
        return (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
    
    def calculate_price_impact(self, lob_data: Dict, trade_size: float, side: str) -> float:
        """
        Calculate price impact for a given trade size
        
        Args:
            lob_data: LOB data
            trade_size: Size of trade
            side: 'buy' or 'sell'
            
        Returns:
            Price impact
        """
        if side == 'buy':
            levels = lob_data.get('asks', [])
        else:
            levels = lob_data.get('bids', [])
        
        if not levels:
            return 0.0
        
        remaining_size = trade_size
        total_cost = 0.0
        
        for level in levels:
            level_size = level['size']
            level_price = level['price']
            
            if remaining_size <= level_size:
                total_cost += remaining_size * level_price
                break
            else:
                total_cost += level_size * level_price
                remaining_size -= level_size
        
        if remaining_size > 0:
            # Not enough liquidity
            return float('inf')
        
        avg_price = total_cost / trade_size
        reference_price = levels[0]['price']
        
        return abs(avg_price - reference_price) / reference_price
    
    def extract_features(self, lob_snapshots: List[Dict], window_size: int = 10) -> np.ndarray:
        """
        Extract features from LOB snapshots
        
        Args:
            lob_snapshots: List of LOB snapshots
            window_size: Window size for feature calculation
            
        Returns:
            Feature array
        """
        features = []
        
        for i in range(len(lob_snapshots)):
            snapshot = lob_snapshots[i]
            processed = self.process_lob_snapshot(snapshot)
            
            # Basic features
            feature_vector = [
                processed['bid_price'],
                processed['ask_price'],
                processed['bid_size'],
                processed['ask_size'],
                processed['spread'],
                processed['mid_price'],
                processed['weighted_mid_price'],
                processed['imbalance'],
                processed['depth_imbalance']
            ]
            
            # Time-based features
            if i >= window_size:
                recent_snapshots = lob_snapshots[i-window_size:i]
                recent_features = [self.process_lob_snapshot(s) for s in recent_snapshots]
                
                # Price volatility
                mid_prices = [f['mid_price'] for f in recent_features if f['mid_price'] > 0]
                if len(mid_prices) > 1:
                    returns = np.diff(np.log(mid_prices))
                    volatility = np.std(returns)
                else:
                    volatility = 0.0
                
                feature_vector.append(volatility)
                
                # Spread volatility
                spreads = [f['spread'] for f in recent_features if f['spread'] > 0]
                if len(spreads) > 1:
                    spread_volatility = np.std(spreads)
                else:
                    spread_volatility = 0.0
                
                feature_vector.append(spread_volatility)
                
                # Volume-weighted features
                total_volume = sum(f['bid_size'] + f['ask_size'] for f in recent_features)
                if total_volume > 0:
                    avg_imbalance = sum(f['imbalance'] * (f['bid_size'] + f['ask_size']) 
                                      for f in recent_features) / total_volume
                else:
                    avg_imbalance = 0.0
                
                feature_vector.append(avg_imbalance)
            else:
                # Pad with zeros for initial snapshots
                feature_vector.extend([0.0, 0.0, 0.0])
            
            features.append(feature_vector)
        
        return np.array(features)


class SyntheticDataGenerator:
    """
    Generator for synthetic market data for training and testing
    """
    
    def __init__(self, base_price: float = 100.0, tick_size: float = 0.01):
        """
        Initialize synthetic data generator
        
        Args:
            base_price: Base price for synthetic data
            tick_size: Minimum price increment
        """
        self.base_price = base_price
        self.tick_size = tick_size
        self.current_price = base_price
        self.time_step = 0
    
    def generate_lob_snapshot(self, depth: int = 10, 
                            volatility: float = 0.001) -> Dict:
        """
        Generate a synthetic LOB snapshot
        
        Args:
            depth: Number of price levels on each side
            volatility: Price volatility
            
        Returns:
            Synthetic LOB snapshot
        """
        # Update price with random walk
        price_change = np.random.normal(0, volatility)
        self.current_price = max(self.current_price + price_change, self.tick_size)
        
        # Generate bid side
        bids = []
        for i in range(depth):
            price = self.current_price - (i + 1) * self.tick_size
            size = np.random.randint(100, 1000)
            bids.append({'price': price, 'size': size})
        
        # Generate ask side
        asks = []
        for i in range(depth):
            price = self.current_price + (i + 1) * self.tick_size
            size = np.random.randint(100, 1000)
            asks.append({'price': price, 'size': size})
        
        self.time_step += 1
        
        return {
            'timestamp': self.time_step,
            'symbol': 'SYNTHETIC',
            'bids': bids,
            'asks': asks,
            'trades': []
        }
    
    def generate_trade(self, lob_snapshot: Dict, side: str = None) -> Dict:
        """
        Generate a synthetic trade
        
        Args:
            lob_snapshot: Current LOB snapshot
            side: Trade side ('buy' or 'sell', random if None)
            
        Returns:
            Synthetic trade
        """
        if side is None:
            side = np.random.choice(['buy', 'sell'])
        
        if side == 'buy' and lob_snapshot['asks']:
            price = lob_snapshot['asks'][0]['price']
            max_size = lob_snapshot['asks'][0]['size']
        elif side == 'sell' and lob_snapshot['bids']:
            price = lob_snapshot['bids'][0]['price']
            max_size = lob_snapshot['bids'][0]['size']
        else:
            return None
        
        size = np.random.randint(1, min(max_size, 500))
        
        return {
            'timestamp': self.time_step,
            'symbol': 'SYNTHETIC',
            'price': price,
            'size': size,
            'side': side,
            'trade_id': f'trade_{self.time_step}_{side}_{size}'
        }
    
    def generate_market_session(self, duration: int = 23400, 
                              update_frequency: float = 1.0) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate a complete market session
        
        Args:
            duration: Session duration in seconds
            update_frequency: Updates per second
            
        Returns:
            Tuple of (LOB snapshots, trades)
        """
        lob_snapshots = []
        trades = []
        
        num_updates = int(duration * update_frequency)
        
        for _ in range(num_updates):
            # Generate LOB snapshot
            snapshot = self.generate_lob_snapshot()
            lob_snapshots.append(snapshot)
            
            # Randomly generate trades
            if np.random.random() < 0.1:  # 10% chance of trade
                trade = self.generate_trade(snapshot)
                if trade:
                    trades.append(trade)
        
        return lob_snapshots, trades
    
    def reset(self, base_price: Optional[float] = None):
        """
        Reset the generator
        
        Args:
            base_price: New base price (optional)
        """
        if base_price is not None:
            self.base_price = base_price
        
        self.current_price = self.base_price
        self.time_step = 0


def load_historical_data(symbol: str, start_date: str, end_date: str, 
                        data_dir: str = 'data/raw') -> pd.DataFrame:
    """
    Load historical market data from files
    
    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        data_dir: Data directory
        
    Returns:
        DataFrame with historical data
    """
    # Placeholder implementation
    # In practice, this would load from saved files
    
    filepath = os.path.join(data_dir, f'{symbol}_{start_date}_{end_date}.csv')
    
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        logger.warning(f"Data file not found: {filepath}")
        return pd.DataFrame()


def save_market_data(data: pd.DataFrame, symbol: str, date: str, 
                    data_dir: str = 'data/processed'):
    """
    Save market data to file
    
    Args:
        data: Market data DataFrame
        symbol: Stock symbol
        date: Date string
        data_dir: Data directory
    """
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = os.path.join(data_dir, f'{symbol}_{date}.csv')
    data.to_csv(filepath, index=False)
    
    logger.info(f"Saved market data to {filepath}")


def preprocess_lob_data(raw_data: List[Dict], processor: LOBDataProcessor) -> np.ndarray:
    """
    Preprocess raw LOB data into features
    
    Args:
        raw_data: Raw LOB data
        processor: LOB data processor
        
    Returns:
        Preprocessed feature array
    """
    return processor.extract_features(raw_data)
