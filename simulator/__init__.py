"""
Simulator package for market making RL environment
"""

from .lob_simulator import LOBSimulator, OrderSide, OrderType, Order, Trade


def __getattr__(name):
    """Load Gym only when the RL environment is requested."""
    if name in {"MarketMakingEnv", "MarketState"}:
        from .market_env import MarketMakingEnv, MarketState
        return {"MarketMakingEnv": MarketMakingEnv, "MarketState": MarketState}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'LOBSimulator',
    'OrderSide', 
    'OrderType',
    'Order',
    'Trade',
    'MarketMakingEnv',
    'MarketState'
]
