"""
Tests for Limit Order Book simulator
"""

import pytest
import numpy as np
import time
from collections import deque

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.lob_simulator import (
    LOBSimulator, LimitOrderBook, Order, Trade, OrderSide, OrderType
)


class TestLimitOrderBook:
    """Test cases for LimitOrderBook class"""
    
    def test_initialization(self):
        """Test LOB initialization"""
        lob = LimitOrderBook("AAPL", tick_size=0.01)
        
        assert lob.symbol == "AAPL"
        assert lob.tick_size == 0.01
        assert len(lob.bids) == 0
        assert len(lob.asks) == 0
        assert lob.get_best_bid() is None
        assert lob.get_best_ask() is None
    
    def test_add_limit_order(self):
        """Test adding limit orders"""
        lob = LimitOrderBook("AAPL", tick_size=0.01)
        
        # Add buy order
        buy_order = Order(
            order_id="buy_1",
            symbol="AAPL",
            side=OrderSide.BUY,
            price=100.0,
            quantity=100,
            timestamp=1.0
        )
        
        trades = lob.add_order(buy_order)
        assert len(trades) == 0  # No matching orders
        assert lob.get_best_bid() == 100.0
        
        # Add sell order
        sell_order = Order(
            order_id="sell_1",
            symbol="AAPL",
            side=OrderSide.SELL,
            price=101.0,
            quantity=100,
            timestamp=2.0
        )
        
        trades = lob.add_order(sell_order)
        assert len(trades) == 0  # No matching orders
        assert lob.get_best_ask() == 101.0
        assert lob.get_spread() == 1.0
        assert lob.get_mid_price() == 100.5
    
    def test_order_matching(self):
        """Test order matching"""
        lob = LimitOrderBook("AAPL", tick_size=0.01)
        
        # Add resting sell order
        sell_order = Order(
            order_id="sell_1",
            symbol="AAPL",
            side=OrderSide.SELL,
            price=100.0,
            quantity=100,
            timestamp=1.0
        )
        lob.add_order(sell_order)
        
        # Add matching buy order
        buy_order = Order(
            order_id="buy_1",
            symbol="AAPL",
            side=OrderSide.BUY,
            price=100.0,
            quantity=50,
            timestamp=2.0
        )
        
        trades = lob.add_order(buy_order)
        
        assert len(trades) == 1
        trade = trades[0]
        assert trade.price == 100.0
        assert trade.quantity == 50
        assert trade.buyer_id == "buy_1"
        assert trade.seller_id == "sell_1"
        
        # Check remaining order
        assert lob.get_best_ask() == 100.0
        assert lob.asks[100.0].total_quantity == 50
    
    def test_order_cancellation(self):
        """Test order cancellation"""
        lob = LimitOrderBook("AAPL", tick_size=0.01)
        
        # Add order
        order = Order(
            order_id="test_1",
            symbol="AAPL",
            side=OrderSide.BUY,
            price=100.0,
            quantity=100,
            timestamp=1.0
        )
        lob.add_order(order)
        
        # Cancel order
        success = lob.cancel_order("test_1")
        assert success == True
        assert lob.get_best_bid() is None
        
        # Try to cancel non-existent order
        success = lob.cancel_order("non_existent")
        assert success == False
    
    def test_market_depth(self):
        """Test market depth functionality"""
        lob = LimitOrderBook("AAPL", tick_size=0.01)
        
        # Add multiple levels
        for i in range(5):
            buy_order = Order(
                order_id=f"buy_{i}",
                symbol="AAPL",
                side=OrderSide.BUY,
                price=100.0 - i * 0.01,
                quantity=100,
                timestamp=1.0
            )
            lob.add_order(buy_order)
            
            sell_order = Order(
                order_id=f"sell_{i}",
                symbol="AAPL",
                side=OrderSide.SELL,
                price=100.01 + i * 0.01,
                quantity=100,
                timestamp=1.0
            )
            lob.add_order(sell_order)
        
        depth = lob.get_market_depth(3)
        
        assert len(depth['bids']) == 3
        assert len(depth['asks']) == 3
        assert depth['bids'][0]['price'] == 100.0
        assert depth['asks'][0]['price'] == 100.01
    
    def test_order_book_imbalance(self):
        """Test order book imbalance calculation"""
        lob = LimitOrderBook("AAPL", tick_size=0.01)
        
        # Add orders with different quantities
        buy_order = Order(
            order_id="buy_1",
            symbol="AAPL",
            side=OrderSide.BUY,
            price=100.0,
            quantity=200,
            timestamp=1.0
        )
        lob.add_order(buy_order)
        
        sell_order = Order(
            order_id="sell_1",
            symbol="AAPL",
            side=OrderSide.SELL,
            price=100.01,
            quantity=100,
            timestamp=1.0
        )
        lob.add_order(sell_order)
        
        imbalance = lob.get_order_book_imbalance()
        expected_imbalance = (200 - 100) / (200 + 100)
        assert abs(imbalance - expected_imbalance) < 1e-10


class TestLOBSimulator:
    """Test cases for LOBSimulator class"""
    
    def test_initialization(self):
        """Test simulator initialization"""
        simulator = LOBSimulator("AAPL", tick_size=0.01)
        
        assert simulator.symbol == "AAPL"
        assert simulator.tick_size == 0.01
        assert simulator.current_time == 0.0
        assert simulator.order_id_counter == 0
    
    def test_step_function(self):
        """Test simulator step function"""
        simulator = LOBSimulator("AAPL", tick_size=0.01)
        
        # Step forward
        state = simulator.step(1.0)
        
        assert state['timestamp'] == 1.0
        assert 'depth' in state
        assert 'mid_price' in state
        assert 'spread' in state
        assert 'imbalance' in state
        assert simulator.current_time == 1.0
    
    def test_place_agent_order(self):
        """Test placing agent orders"""
        simulator = LOBSimulator("AAPL", tick_size=0.01)
        
        # Place buy order
        order_id = simulator.place_agent_order(
            side="buy",
            price=100.0,
            quantity=100
        )
        
        assert order_id.startswith("agent_")
        assert simulator.order_id_counter == 1
        assert order_id in simulator.agent_orders
        
        # Place sell order
        order_id2 = simulator.place_agent_order(
            side="sell",
            price=101.0,
            quantity=100
        )
        
        assert order_id2.startswith("agent_")
        assert simulator.order_id_counter == 2
        assert order_id2 in simulator.agent_orders
    
    def test_cancel_agent_order(self):
        """Test canceling agent orders"""
        simulator = LOBSimulator("AAPL", tick_size=0.01)
        
        # Place order
        order_id = simulator.place_agent_order(
            side="buy",
            price=100.0,
            quantity=100
        )
        
        # Cancel order
        success = simulator.cancel_agent_order(order_id)
        assert success == True
        assert order_id not in simulator.agent_orders
        
        # Try to cancel non-existent order
        success = simulator.cancel_agent_order("non_existent")
        assert success == False
    
    def test_reset(self):
        """Test simulator reset"""
        simulator = LOBSimulator("AAPL", tick_size=0.01)
        
        # Place some orders and advance time
        simulator.place_agent_order("buy", 100.0, 100)
        simulator.step(10.0)
        
        # Reset
        simulator.reset()
        
        assert simulator.current_time == 0.0
        assert simulator.order_id_counter == 0
        assert len(simulator.agent_orders) == 0
        assert len(simulator.lob.orders) == 0


class TestOrder:
    """Test cases for Order class"""
    
    def test_order_creation(self):
        """Test order creation"""
        order = Order(
            order_id="test_1",
            symbol="AAPL",
            side=OrderSide.BUY,
            price=100.0,
            quantity=100,
            timestamp=1.0,
            ttl=10.0
        )
        
        assert order.order_id == "test_1"
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.price == 100.0
        assert order.quantity == 100
        assert order.timestamp == 1.0
        assert order.ttl == 10.0
        assert order.expiry_time == 11.0
    
    def test_order_expiry(self):
        """Test order expiry"""
        order = Order(
            order_id="test_1",
            symbol="AAPL",
            side=OrderSide.BUY,
            price=100.0,
            quantity=100,
            timestamp=1.0,
            ttl=10.0
        )
        
        # Not expired
        assert order.is_expired(5.0) == False
        
        # Expired
        assert order.is_expired(12.0) == True
        
        # Order without TTL
        order_no_ttl = Order(
            order_id="test_2",
            symbol="AAPL",
            side=OrderSide.BUY,
            price=100.0,
            quantity=100,
            timestamp=1.0
        )
        
        assert order_no_ttl.is_expired(100.0) == False


if __name__ == "__main__":
    pytest.main([__file__])
