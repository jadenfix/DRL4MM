"""
Logging utilities for the market making RL project
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(name: str, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """
    Setup a logger with both file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Configured logger
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """
    Specialized logger for training metrics and progress
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initialize training logger
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup main logger
        self.logger = setup_logger(
            name=f"training_{experiment_name}",
            log_file=os.path.join(log_dir, f"{experiment_name}.log"),
            level="INFO"
        )
        
        # Setup metrics file
        self.metrics_file = os.path.join(log_dir, f"{experiment_name}_metrics.csv")
        self._init_metrics_file()
    
    def _init_metrics_file(self):
        """Initialize CSV file for metrics"""
        with open(self.metrics_file, 'w') as f:
            f.write("timestamp,episode,step,reward,pnl,inventory,trades,spread,mid_price\n")
    
    def log_episode(self, episode: int, reward: float, pnl: float, 
                   inventory: int, trades: int, steps: int):
        """
        Log episode results
        
        Args:
            episode: Episode number
            reward: Episode reward
            pnl: Episode PnL
            inventory: Final inventory
            trades: Number of trades
            steps: Number of steps
        """
        self.logger.info(
            f"Episode {episode:6d} | "
            f"Reward: {reward:8.2f} | "
            f"PnL: {pnl:8.2f} | "
            f"Inventory: {inventory:6d} | "
            f"Trades: {trades:4d} | "
            f"Steps: {steps:6d}"
        )
    
    def log_training_step(self, episode: int, step: int, actor_loss: float, 
                         critic_loss: float, exploration_noise: float):
        """
        Log training step details
        
        Args:
            episode: Episode number
            step: Step number
            actor_loss: Actor loss
            critic_loss: Critic loss
            exploration_noise: Current exploration noise
        """
        if step % 1000 == 0:  # Log every 1000 steps
            self.logger.info(
                f"Episode {episode:6d} Step {step:6d} | "
                f"Actor Loss: {actor_loss:8.4f} | "
                f"Critic Loss: {critic_loss:8.4f} | "
                f"Noise: {exploration_noise:6.4f}"
            )
    
    def log_evaluation(self, episode: int, eval_reward: float, eval_pnl: float,
                      eval_trades: float, eval_inventory: float):
        """
        Log evaluation results
        
        Args:
            episode: Episode number
            eval_reward: Average evaluation reward
            eval_pnl: Average evaluation PnL
            eval_trades: Average evaluation trades
            eval_inventory: Average evaluation inventory
        """
        self.logger.info(
            f"EVAL Episode {episode:6d} | "
            f"Reward: {eval_reward:8.2f} | "
            f"PnL: {eval_pnl:8.2f} | "
            f"Trades: {eval_trades:6.2f} | "
            f"Inventory: {eval_inventory:6.2f}"
        )
    
    def log_metrics_csv(self, timestamp: str, episode: int, step: int, 
                       reward: float, pnl: float, inventory: int, trades: int,
                       spread: float, mid_price: float):
        """
        Log metrics to CSV file
        
        Args:
            timestamp: Timestamp
            episode: Episode number
            step: Step number
            reward: Reward
            pnl: PnL
            inventory: Inventory
            trades: Number of trades
            spread: Spread
            mid_price: Mid price
        """
        with open(self.metrics_file, 'a') as f:
            f.write(f"{timestamp},{episode},{step},{reward},{pnl},{inventory},"
                   f"{trades},{spread},{mid_price}\n")
    
    def log_error(self, message: str, exception: Exception = None):
        """
        Log error message
        
        Args:
            message: Error message
            exception: Exception object (optional)
        """
        if exception:
            self.logger.error(f"{message}: {str(exception)}", exc_info=True)
        else:
            self.logger.error(message)
    
    def log_warning(self, message: str):
        """
        Log warning message
        
        Args:
            message: Warning message
        """
        self.logger.warning(message)
    
    def log_info(self, message: str):
        """
        Log info message
        
        Args:
            message: Info message
        """
        self.logger.info(message)
