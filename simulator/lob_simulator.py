"""
Limit Order Book (LOB) Simulator for Market Making RL Environment

This module implements a high-fidelity matching engine with FIFO order processing,
realistic fill logic, and integration with real Nasdaq market data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import heapq
import time
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration"""
    LIMIT = "limit"
    MARKET = "market"
    CANCEL = "cancel"


@dataclass
class Order:
    """
    Represents a single order in the limit order book
    """
    order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: int
    timestamp: float
    order_type: OrderType = OrderType.LIMIT
    ttl: Optional[float] = None  # Time to live in seconds
    agent_id: Optional[str] = None  # For tracking agent orders
    
    def __post_init__(self):
        if self.ttl is not None:
            self.expiry_time = self.timestamp + self.ttl
        else:
            self.expiry_time = None
    
    def is_expired(self, current_time: float) -> bool:
        """Check if order has expired"""
        if self.expiry_time is None:
            return False
        return current_time >= self.expiry_time


@dataclass
class Trade:
    """
    Represents a completed trade
    """
    trade_id: str
    symbol: str
    price: float
    quantity: int
    timestamp: float
    buyer_id: str
    seller_id: str
    aggressive_side: OrderSide


@dataclass
class OrderBookLevel:
    """
    Represents a single price level in the order book
    """
    price: float
    total_quantity: int
    order_count: int
    orders: deque  # FIFO queue of orders at this price level
    
    def __post_init__(self):
        if not hasattr(self, 'orders'):
            self.orders = deque()


class LimitOrderBook:
    """
    High-performance FIFO limit order book implementation
    """
    
    def __init__(self, symbol: str, tick_size: float = 0.01):
        self.symbol = symbol
        self.tick_size = tick_size
        
        # Order book data structures
        self.bids = {}  # price -> OrderBookLevel
        self.asks = {}  # price -> OrderBookLevel
        
        # Sorted price levels for efficient access
        self.bid_prices = []  # Max heap (negative prices)
        self.ask_prices = []  # Min heap
        
        # Order tracking
        self.orders = {}  # order_id -> Order
        self.order_to_level = {}  # order_id -> (side, price)
        
        # Market data
        self.last_trade_price = None
        self.last_trade_time = None
        self.trades = deque(maxlen=1000)  # Recent trades
        
        # Statistics
        self.total_volume = 0
        self.trade_count = 0
        
    def add_order(self, order: Order) -> List[Trade]:
        """
        Add order to the book and return any resulting trades
        """
        if order.order_type == OrderType.MARKET:
            return self._process_market_order(order)
        elif order.order_type == OrderType.LIMIT:
            return self._process_limit_order(order)
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order by ID
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        side, price = self.order_to_level[order_id]
        
        # Remove from appropriate side
        if side == OrderSide.BUY:
            level = self.bids[price]
        else:
            level = self.asks[price]
        
        # Remove order from level
        level.orders.remove(order)
        level.total_quantity -= order.quantity
        level.order_count -= 1
        
        # Clean up empty levels
        if level.order_count == 0:
            if side == OrderSide.BUY:
                del self.bids[price]
                self.bid_prices.remove(-price)
                heapq.heapify(self.bid_prices)
            else:
                del self.asks[price]
                self.ask_prices.remove(price)
                heapq.heapify(self.ask_prices)
        
        # Clean up tracking
        del self.orders[order_id]
        del self.order_to_level[order_id]
        
        return True
    
    def _process_market_order(self, order: Order) -> List[Trade]:
        """
        Process a market order against the book
        """
        trades = []
        remaining_qty = order.quantity
        
        if order.side == OrderSide.BUY:
            # Buy market order - match against asks
            while remaining_qty > 0 and self.ask_prices:
                best_ask_price = self.ask_prices[0]
                level = self.asks[best_ask_price]
                
                trade, consumed_qty = self._execute_trade_at_level(
                    order, level, best_ask_price, remaining_qty
                )
                
                if trade:
                    trades.append(trade)
                    remaining_qty -= consumed_qty
                
                # Remove empty level
                if level.order_count == 0:
                    del self.asks[best_ask_price]
                    heapq.heappop(self.ask_prices)
        
        else:
            # Sell market order - match against bids
            while remaining_qty > 0 and self.bid_prices:
                best_bid_price = -self.bid_prices[0]
                level = self.bids[best_bid_price]
                
                trade, consumed_qty = self._execute_trade_at_level(
                    order, level, best_bid_price, remaining_qty
                )
                
                if trade:
                    trades.append(trade)
                    remaining_qty -= consumed_qty
                
                # Remove empty level
                if level.order_count == 0:
                    del self.bids[best_bid_price]
                    heapq.heappop(self.bid_prices)
        
        return trades
    
    def _process_limit_order(self, order: Order) -> List[Trade]:
        """
        Process a limit order - first try to match, then add to book
        """
        trades = []
        remaining_qty = order.quantity
        
        # Try to match against existing orders
        if order.side == OrderSide.BUY:
            # Buy limit order - match against asks at or below limit price
            while remaining_qty > 0 and self.ask_prices:
                best_ask_price = self.ask_prices[0]
                if best_ask_price <= order.price:
                    level = self.asks[best_ask_price]
                    
                    trade, consumed_qty = self._execute_trade_at_level(
                        order, level, best_ask_price, remaining_qty
                    )
                    
                    if trade:
                        trades.append(trade)
                        remaining_qty -= consumed_qty
                    
                    # Remove empty level
                    if level.order_count == 0:
                        del self.asks[best_ask_price]
                        heapq.heappop(self.ask_prices)
                else:
                    break
        
        else:
            # Sell limit order - match against bids at or above limit price
            while remaining_qty > 0 and self.bid_prices:
                best_bid_price = -self.bid_prices[0]
                if best_bid_price >= order.price:
                    level = self.bids[best_bid_price]
                    
                    trade, consumed_qty = self._execute_trade_at_level(
                        order, level, best_bid_price, remaining_qty
                    )
                    
                    if trade:
                        trades.append(trade)
                        remaining_qty -= consumed_qty
                    
                    # Remove empty level
                    if level.order_count == 0:
                        del self.bids[best_bid_price]
                        heapq.heappop(self.bid_prices)
                else:
                    break
        
        # Add remaining quantity to book if any
        if remaining_qty > 0:
            remaining_order = Order(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                price=order.price,
                quantity=remaining_qty,
                timestamp=order.timestamp,
                order_type=order.order_type,
                ttl=order.ttl,
                agent_id=order.agent_id
            )
            self._add_order_to_book(remaining_order)
        
        return trades
    
    def _execute_trade_at_level(self, incoming_order: Order, level: OrderBookLevel, 
                               price: float, max_qty: int) -> Tuple[Optional[Trade], int]:
        """
        Execute trade at a specific price level
        """
        if level.order_count == 0:
            return None, 0
        
        # Get first order in queue (FIFO)
        resting_order = level.orders[0]
        
        # Calculate trade quantity
        trade_qty = min(max_qty, resting_order.quantity)
        
        # Create trade
        trade = Trade(
            trade_id=f"{self.symbol}_{int(time.time() * 1000000)}",
            symbol=self.symbol,
            price=price,
            quantity=trade_qty,
            timestamp=incoming_order.timestamp,
            buyer_id=incoming_order.order_id if incoming_order.side == OrderSide.BUY else resting_order.order_id,
            seller_id=resting_order.order_id if incoming_order.side == OrderSide.BUY else incoming_order.order_id,
            aggressive_side=incoming_order.side
        )
        
        # Update statistics
        self.last_trade_price = price
        self.last_trade_time = incoming_order.timestamp
        self.total_volume += trade_qty
        self.trade_count += 1
        self.trades.append(trade)
        
        # Update resting order
        resting_order.quantity -= trade_qty
        level.total_quantity -= trade_qty
        
        # Remove order if fully filled
        if resting_order.quantity == 0:
            level.orders.popleft()
            level.order_count -= 1
            del self.orders[resting_order.order_id]
            del self.order_to_level[resting_order.order_id]
        
        return trade, trade_qty
    
    def _add_order_to_book(self, order: Order):
        """
        Add order to the appropriate side of the book
        """
        price = order.price
        
        if order.side == OrderSide.BUY:
            # Add to bids
            if price not in self.bids:
                self.bids[price] = OrderBookLevel(price, 0, 0, deque())
                heapq.heappush(self.bid_prices, -price)  # Max heap
            
            level = self.bids[price]
        else:
            # Add to asks
            if price not in self.asks:
                self.asks[price] = OrderBookLevel(price, 0, 0, deque())
                heapq.heappush(self.ask_prices, price)  # Min heap
            
            level = self.asks[price]
        
        # Add order to level
        level.orders.append(order)
        level.total_quantity += order.quantity
        level.order_count += 1
        
        # Track order
        self.orders[order.order_id] = order
        self.order_to_level[order.order_id] = (order.side, price)
    
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price"""
        if not self.bid_prices:
            return None
        return -self.bid_prices[0]
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price"""
        if not self.ask_prices:
            return None
        return self.ask_prices[0]
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return best_ask - best_bid
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is None or best_ask is None:
            return self.last_trade_price
        return (best_bid + best_ask) / 2
    
    def get_market_depth(self, levels: int = 10) -> Dict:
        """
        Get market depth up to specified levels
        """
        bids = []
        asks = []
        
        # Get top bid levels
        sorted_bids = sorted(self.bids.keys(), reverse=True)
        for i, price in enumerate(sorted_bids[:levels]):
            level = self.bids[price]
            bids.append({
                'price': price,
                'quantity': level.total_quantity,
                'orders': level.order_count
            })
        
        # Get top ask levels
        sorted_asks = sorted(self.asks.keys())
        for i, price in enumerate(sorted_asks[:levels]):
            level = self.asks[price]
            asks.append({
                'price': price,
                'quantity': level.total_quantity,
                'orders': level.order_count
            })
        
        return {
            'bids': bids,
            'asks': asks,
            'timestamp': time.time()
        }
    
    def get_order_book_imbalance(self) -> float:
        """
        Calculate order book imbalance
        """
        total_bid_volume = sum(level.total_quantity for level in self.bids.values())
        total_ask_volume = sum(level.total_quantity for level in self.asks.values())
        
        if total_bid_volume + total_ask_volume == 0:
            return 0.0
        
        return (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
    
    def cleanup_expired_orders(self, current_time: float) -> List[str]:
        """
        Remove expired orders from the book
        """
        expired_orders = []
        
        for order_id, order in list(self.orders.items()):
            if order.is_expired(current_time):
                if self.cancel_order(order_id):
                    expired_orders.append(order_id)
        
        return expired_orders


class LOBSimulator:
    """
    Main simulator class that manages the order book and provides
    the interface for the RL environment
    """
    
    def __init__(self, symbol: str, tick_size: float = 0.01, 
                 latency_mean: float = 0.001, latency_std: float = 0.0005):
        self.symbol = symbol
        self.tick_size = tick_size
        self.latency_mean = latency_mean
        self.latency_std = latency_std
        
        # Initialize order book
        self.lob = LimitOrderBook(symbol, tick_size)
        
        # Simulation state
        self.current_time = 0.0
        self.order_id_counter = 0
        self.agent_orders = {}  # track agent orders
        
        # Performance tracking
        self.total_trades = 0
        self.total_volume = 0
        
        logger.info(f"Initialized LOB simulator for {symbol}")
    
    def step(self, timestamp: float) -> Dict:
        """
        Advance simulation by one time step
        """
        self.current_time = timestamp
        
        # Clean up expired orders
        expired_orders = self.lob.cleanup_expired_orders(timestamp)
        
        # Return current state
        return {
            'timestamp': timestamp,
            'depth': self.lob.get_market_depth(),
            'mid_price': self.lob.get_mid_price(),
            'spread': self.lob.get_spread(),
            'imbalance': self.lob.get_order_book_imbalance(),
            'expired_orders': expired_orders,
            'total_trades': self.lob.trade_count,
            'total_volume': self.lob.total_volume
        }
    
    def place_agent_order(self, side: str, price: float, quantity: int, 
                         agent_id: str = "agent", ttl: Optional[float] = None) -> str:
        """
        Place order for RL agent
        """
        # Generate order ID
        order_id = f"{agent_id}_{self.order_id_counter}"
        self.order_id_counter += 1
        
        # Create order
        order = Order(
            order_id=order_id,
            symbol=self.symbol,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            price=price,
            quantity=quantity,
            timestamp=self.current_time,
            order_type=OrderType.LIMIT,
            ttl=ttl,
            agent_id=agent_id
        )
        
        # Add latency simulation
        if self.latency_mean > 0:
            latency = np.random.normal(self.latency_mean, self.latency_std)
            order.timestamp += max(0, latency)
        
        # Process order
        trades = self.lob.add_order(order)
        
        # Track agent order
        if order_id in self.lob.orders:
            self.agent_orders[order_id] = order
        
        return order_id
    
    def cancel_agent_order(self, order_id: str) -> bool:
        """
        Cancel agent order
        """
        success = self.lob.cancel_order(order_id)
        if success and order_id in self.agent_orders:
            del self.agent_orders[order_id]
        return success
    
    def get_agent_fills(self, agent_id: str = "agent") -> List[Trade]:
        """
        Get recent fills for agent
        """
        agent_fills = []
        for trade in self.lob.trades:
            if (trade.buyer_id.startswith(agent_id) or 
                trade.seller_id.startswith(agent_id)):
                agent_fills.append(trade)
        return agent_fills
    
    def reset(self):
        """
        Reset simulator state
        """
        self.lob = LimitOrderBook(self.symbol, self.tick_size)
        self.current_time = 0.0
        self.order_id_counter = 0
        self.agent_orders = {}
        logger.info(f"Reset LOB simulator for {self.symbol}")
