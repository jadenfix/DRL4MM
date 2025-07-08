"""
Metrics tracking and evaluation utilities for market making RL
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


class MetricsTracker:
    """
    Comprehensive metrics tracker for market making RL training
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker
        
        Args:
            window_size: Window size for moving averages
        """
        self.window_size = window_size
        
        # Episode metrics
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_pnls = deque(maxlen=window_size)
        self.episode_trades = deque(maxlen=window_size)
        self.episode_inventories = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        
        # Training metrics
        self.actor_losses = deque(maxlen=window_size * 10)
        self.critic_losses = deque(maxlen=window_size * 10)
        self.exploration_noise = deque(maxlen=window_size * 10)
        
        # Market metrics
        self.spreads = deque(maxlen=window_size * 10)
        self.mid_prices = deque(maxlen=window_size * 10)
        self.volumes = deque(maxlen=window_size * 10)
        
        # Evaluation metrics
        self.eval_rewards = []
        self.eval_pnls = []
        self.eval_trades = []
        self.eval_episodes = []
        
        # Raw data for detailed analysis
        self.raw_data = defaultdict(list)
    
    def log_episode_metrics(self, metrics: Dict, episode: int):
        """
        Log episode-level metrics
        
        Args:
            metrics: Dictionary of episode metrics
            episode: Episode number
        """
        self.episode_rewards.append(metrics.get('episode_reward', 0))
        self.episode_pnls.append(metrics.get('episode_pnl', 0))
        self.episode_trades.append(metrics.get('episode_trades', 0))
        self.episode_inventories.append(metrics.get('final_inventory', 0))
        self.episode_lengths.append(metrics.get('episode_steps', 0))
        
        # Store raw data
        for key, value in metrics.items():
            self.raw_data[key].append(value)
        
        self.raw_data['episode'].append(episode)
        self.raw_data['timestamp'].append(datetime.now().isoformat())
    
    def log_training_metrics(self, metrics: Dict, episode: int, step: int):
        """
        Log training step metrics
        
        Args:
            metrics: Dictionary of training metrics
            episode: Episode number
            step: Step number
        """
        if 'actor_loss' in metrics:
            self.actor_losses.append(metrics['actor_loss'])
        
        if 'critic_loss' in metrics:
            self.critic_losses.append(metrics['critic_loss'])
        
        if 'exploration_noise' in metrics:
            self.exploration_noise.append(metrics['exploration_noise'])
    
    def log_market_metrics(self, spread: float, mid_price: float, volume: float):
        """
        Log market-level metrics
        
        Args:
            spread: Bid-ask spread
            mid_price: Mid price
            volume: Volume
        """
        self.spreads.append(spread)
        self.mid_prices.append(mid_price)
        self.volumes.append(volume)
    
    def log_evaluation_metrics(self, metrics: Dict, episode: int):
        """
        Log evaluation metrics
        
        Args:
            metrics: Dictionary of evaluation metrics
            episode: Episode number
        """
        self.eval_rewards.append(metrics.get('eval_reward_mean', 0))
        self.eval_pnls.append(metrics.get('eval_pnl_mean', 0))
        self.eval_trades.append(metrics.get('eval_trades_mean', 0))
        self.eval_episodes.append(episode)
    
    def get_episode_stats(self) -> Dict:
        """
        Get episode-level statistics
        
        Returns:
            Dictionary of episode statistics
        """
        if not self.episode_rewards:
            return {}
        
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'mean_pnl': np.mean(self.episode_pnls),
            'std_pnl': np.std(self.episode_pnls),
            'mean_trades': np.mean(self.episode_trades),
            'mean_inventory': np.mean(np.abs(self.episode_inventories)),
            'mean_episode_length': np.mean(self.episode_lengths)
        }
    
    def get_training_stats(self) -> Dict:
        """
        Get training statistics
        
        Returns:
            Dictionary of training statistics
        """
        stats = {}
        
        if self.actor_losses:
            stats['mean_actor_loss'] = np.mean(self.actor_losses)
            stats['std_actor_loss'] = np.std(self.actor_losses)
        
        if self.critic_losses:
            stats['mean_critic_loss'] = np.mean(self.critic_losses)
            stats['std_critic_loss'] = np.std(self.critic_losses)
        
        if self.exploration_noise:
            stats['current_exploration_noise'] = self.exploration_noise[-1]
        
        return stats
    
    def get_market_stats(self) -> Dict:
        """
        Get market statistics
        
        Returns:
            Dictionary of market statistics
        """
        stats = {}
        
        if self.spreads:
            stats['mean_spread'] = np.mean(self.spreads)
            stats['std_spread'] = np.std(self.spreads)
        
        if self.mid_prices:
            stats['mean_mid_price'] = np.mean(self.mid_prices)
            stats['price_volatility'] = np.std(self.mid_prices)
        
        if self.volumes:
            stats['mean_volume'] = np.mean(self.volumes)
            stats['std_volume'] = np.std(self.volumes)
        
        return stats
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio of episode rewards
        
        Args:
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sharpe ratio
        """
        if len(self.episode_rewards) < 2:
            return 0.0
        
        returns = np.array(self.episode_rewards)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Assuming daily returns, convert risk-free rate
        daily_rf = risk_free_rate / 252
        sharpe = (mean_return - daily_rf) / std_return
        
        return sharpe
    
    def calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown of cumulative PnL
        
        Returns:
            Maximum drawdown
        """
        if not self.episode_pnls:
            return 0.0
        
        cumulative_pnl = np.cumsum(self.episode_pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        
        return np.max(drawdown)
    
    def calculate_hit_ratio(self) -> float:
        """
        Calculate hit ratio (proportion of profitable episodes)
        
        Returns:
            Hit ratio
        """
        if not self.episode_pnls:
            return 0.0
        
        profitable_episodes = sum(1 for pnl in self.episode_pnls if pnl > 0)
        return profitable_episodes / len(self.episode_pnls)
    
    def get_comprehensive_stats(self) -> Dict:
        """
        Get comprehensive statistics
        
        Returns:
            Dictionary of all statistics
        """
        stats = {}
        
        # Episode stats
        stats.update(self.get_episode_stats())
        
        # Training stats
        stats.update(self.get_training_stats())
        
        # Market stats
        stats.update(self.get_market_stats())
        
        # Risk metrics
        stats['sharpe_ratio'] = self.calculate_sharpe_ratio()
        stats['max_drawdown'] = self.calculate_max_drawdown()
        stats['hit_ratio'] = self.calculate_hit_ratio()
        
        return stats
    
    def save_metrics(self, filepath: str):
        """
        Save metrics to CSV file
        
        Args:
            filepath: Path to save CSV file
        """
        # Create DataFrame from raw data
        df = pd.DataFrame(self.raw_data)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
    
    def plot_training_progress(self, save_path: Optional[str] = None):
        """
        Plot training progress
        
        Args:
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        if self.episode_rewards:
            axes[0, 0].plot(list(self.episode_rewards))
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
        
        # Episode PnL
        if self.episode_pnls:
            axes[0, 1].plot(list(self.episode_pnls))
            axes[0, 1].set_title('Episode PnL')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('PnL')
        
        # Training losses
        if self.actor_losses and self.critic_losses:
            axes[1, 0].plot(list(self.actor_losses), label='Actor Loss', alpha=0.7)
            axes[1, 0].plot(list(self.critic_losses), label='Critic Loss', alpha=0.7)
            axes[1, 0].set_title('Training Losses')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
        
        # Episode trades
        if self.episode_trades:
            axes[1, 1].plot(list(self.episode_trades))
            axes[1, 1].set_title('Episode Trades')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Number of Trades')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_evaluation_progress(self, save_path: Optional[str] = None):
        """
        Plot evaluation progress
        
        Args:
            save_path: Path to save plot (optional)
        """
        if not self.eval_rewards:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Evaluation rewards
        axes[0].plot(self.eval_episodes, self.eval_rewards)
        axes[0].set_title('Evaluation Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Average Reward')
        
        # Evaluation PnL
        axes[1].plot(self.eval_episodes, self.eval_pnls)
        axes[1].set_title('Evaluation PnL')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Average PnL')
        
        # Evaluation trades
        axes[2].plot(self.eval_episodes, self.eval_trades)
        axes[2].set_title('Evaluation Trades')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Average Trades')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def print_summary(self):
        """
        Print summary statistics
        """
        stats = self.get_comprehensive_stats()
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        
        print(f"Episodes: {len(self.episode_rewards)}")
        print(f"Mean Reward: {stats.get('mean_reward', 0):.2f} ± {stats.get('std_reward', 0):.2f}")
        print(f"Mean PnL: {stats.get('mean_pnl', 0):.2f} ± {stats.get('std_pnl', 0):.2f}")
        print(f"Mean Trades: {stats.get('mean_trades', 0):.2f}")
        print(f"Mean Inventory: {stats.get('mean_inventory', 0):.2f}")
        print(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.4f}")
        print(f"Max Drawdown: {stats.get('max_drawdown', 0):.2f}")
        print(f"Hit Ratio: {stats.get('hit_ratio', 0):.4f}")
        
        if stats.get('mean_actor_loss'):
            print(f"Mean Actor Loss: {stats.get('mean_actor_loss', 0):.6f}")
        if stats.get('mean_critic_loss'):
            print(f"Mean Critic Loss: {stats.get('mean_critic_loss', 0):.6f}")
        
        print("="*50)


class PerformanceAnalyzer:
    """
    Advanced performance analysis for market making strategies
    """
    
    def __init__(self, metrics_tracker: MetricsTracker):
        """
        Initialize performance analyzer
        
        Args:
            metrics_tracker: MetricsTracker instance
        """
        self.metrics = metrics_tracker
    
    def analyze_inventory_management(self) -> Dict:
        """
        Analyze inventory management performance
        
        Returns:
            Dictionary of inventory analysis
        """
        if not self.metrics.episode_inventories:
            return {}
        
        inventories = np.array(self.metrics.episode_inventories)
        
        return {
            'inventory_mean': np.mean(inventories),
            'inventory_std': np.std(inventories),
            'inventory_max': np.max(np.abs(inventories)),
            'inventory_turnover': np.mean(np.abs(np.diff(inventories))) if len(inventories) > 1 else 0,
            'inventory_violations': np.sum(np.abs(inventories) > 500)  # Assuming 500 is a threshold
        }
    
    def analyze_trading_patterns(self) -> Dict:
        """
        Analyze trading patterns and execution quality
        
        Returns:
            Dictionary of trading analysis
        """
        if not self.metrics.episode_trades:
            return {}
        
        trades = np.array(self.metrics.episode_trades)
        
        return {
            'avg_trades_per_episode': np.mean(trades),
            'trade_frequency_std': np.std(trades),
            'max_trades_episode': np.max(trades),
            'min_trades_episode': np.min(trades),
            'zero_trade_episodes': np.sum(trades == 0)
        }
    
    def analyze_risk_metrics(self) -> Dict:
        """
        Analyze risk metrics
        
        Returns:
            Dictionary of risk analysis
        """
        if not self.metrics.episode_pnls:
            return {}
        
        pnls = np.array(self.metrics.episode_pnls)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(pnls, 5)
        var_99 = np.percentile(pnls, 1)
        
        # Conditional VaR (CVaR)
        cvar_95 = np.mean(pnls[pnls <= var_95])
        cvar_99 = np.mean(pnls[pnls <= var_99])
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'volatility': np.std(pnls),
            'skewness': self._calculate_skewness(pnls),
            'kurtosis': self._calculate_kurtosis(pnls)
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive performance report
        
        Args:
            save_path: Path to save report (optional)
            
        Returns:
            Report string
        """
        report = []
        report.append("MARKET MAKING PERFORMANCE REPORT")
        report.append("=" * 50)
        
        # Basic statistics
        basic_stats = self.metrics.get_comprehensive_stats()
        report.append("\nBASIC STATISTICS:")
        for key, value in basic_stats.items():
            report.append(f"{key}: {value:.4f}")
        
        # Inventory analysis
        inventory_stats = self.analyze_inventory_management()
        if inventory_stats:
            report.append("\nINVENTORY MANAGEMENT:")
            for key, value in inventory_stats.items():
                report.append(f"{key}: {value:.4f}")
        
        # Trading patterns
        trading_stats = self.analyze_trading_patterns()
        if trading_stats:
            report.append("\nTRADING PATTERNS:")
            for key, value in trading_stats.items():
                report.append(f"{key}: {value:.4f}")
        
        # Risk metrics
        risk_stats = self.analyze_risk_metrics()
        if risk_stats:
            report.append("\nRISK METRICS:")
            for key, value in risk_stats.items():
                report.append(f"{key}: {value:.4f}")
        
        report_str = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)
        
        return report_str
