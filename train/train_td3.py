"""
Training Script for TD3 Market Making Agent

This script handles the main training loop for the TD3 agent in the market making environment.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import wandb
from datetime import datetime
import logging
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.market_env import MarketMakingEnv
from agent.td3_agent import TD3Agent
from utils.logger import setup_logger
from utils.metrics import MetricsTracker


def setup_logging(config: Dict) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = os.path.join(config['logging']['log_dir'], 
                          f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = setup_logger(
        name="td3_training",
        log_file=os.path.join(log_dir, "training.log"),
        level=config['logging']['level']
    )
    
    return logger


def setup_wandb(config: Dict) -> None:
    """Setup Weights & Biases logging"""
    if config['training']['use_wandb']:
        wandb.init(
            project=config['training']['wandb_project'],
            entity=config['training']['wandb_entity'],
            name=config['training']['experiment_name'],
            config=config
        )


def create_environment(config: Dict) -> MarketMakingEnv:
    """Create and configure the market making environment"""
    env_config = {
        'max_inventory': config['environment']['max_inventory'],
        'inventory_penalty': config['environment']['inventory_penalty'],
        'tick_size': config['environment']['tick_size'],
        'min_spread': config['environment']['min_spread'],
        'max_spread': config['environment']['max_spread'],
        'quote_ttl': config['environment'].get('quote_ttl', 10.0),
        'latency_mean': config['simulation']['latency_mean'],
        'latency_std': config['simulation']['latency_std'],
        'lob_depth': config['data']['lob_depth'],
        'feature_window': config['data']['feature_window'],
        'max_episode_steps': config['environment']['max_episode_steps'],
        'reward_scaling': config['environment']['reward_config']['pnl_weight'],
        'fill_bonus': config['environment']['reward_config']['fill_bonus'],
        'spread_penalty': config['environment']['reward_config']['spread_penalty']
    }
    
    # Create environment
    env = MarketMakingEnv(
        symbol=config['data']['symbols'][0],  # Use first symbol for training
        config=env_config,
        data_source="synthetic"  # Start with synthetic data
    )
    
    return env


def create_agent(env: MarketMakingEnv, config: Dict) -> TD3Agent:
    """Create and configure the TD3 agent"""
    agent_config = {
        'actor_lr': config['agent']['learning_rate'],
        'critic_lr': config['agent']['learning_rate'],
        'buffer_size': config['agent']['buffer_size'],
        'batch_size': config['agent']['batch_size'],
        'gamma': 0.99,
        'tau': config['agent']['target_update_rate'],
        'policy_noise': config['agent']['target_policy_noise'],
        'noise_clip': config['agent']['policy_noise_clip'],
        'policy_freq': config['agent']['policy_delay'],
        'exploration_noise': config['agent']['exploration_noise'],
        'noise_decay': 0.995,
        'min_noise': 0.01,
        'use_cuda': torch.cuda.is_available(),
        'hidden_dims': config['agent']['hidden_dims'],
        'activation': config['agent']['activation']
    }
    
    agent = TD3Agent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=float(env.action_space.high[0]),
        config=agent_config
    )
    
    return agent


def train_episode(env: MarketMakingEnv, agent: TD3Agent, metrics: MetricsTracker, 
                 episode: int, logger: logging.Logger) -> Dict:
    """Train for one episode"""
    
    # Reset environment
    state = env.reset()
    
    episode_reward = 0
    episode_steps = 0
    episode_trades = 0
    episode_pnl = 0
    
    done = False
    
    while not done:
        # Select action
        action = agent.select_action(state, add_noise=True)
        
        # Take step in environment
        next_state, reward, done, info = env.step(action)
        
        # Store transition
        agent.store_transition(state, action, reward, next_state, done)
        
        # Train agent
        if len(agent.replay_buffer) >= agent.batch_size:
            train_metrics = agent.train()
            
            # Log training metrics
            if train_metrics and episode_steps % 100 == 0:
                metrics.log_training_metrics(train_metrics, episode, episode_steps)
        
        # Update state
        state = next_state
        episode_reward += reward
        episode_steps += 1
        
        # Track episode metrics
        episode_trades = info.get('fill_count', 0)
        episode_pnl = info.get('realized_pnl', 0) + info.get('unrealized_pnl', 0)
    
    # Log episode completion
    logger.info(f"Episode {episode} completed: "
               f"Reward={episode_reward:.2f}, "
               f"Steps={episode_steps}, "
               f"Trades={episode_trades}, "
               f"PnL={episode_pnl:.2f}")
    
    # Add episode reward to agent
    agent.add_episode_reward(episode_reward)
    
    return {
        'episode_reward': episode_reward,
        'episode_steps': episode_steps,
        'episode_trades': episode_trades,
        'episode_pnl': episode_pnl,
        'final_inventory': info.get('inventory', 0),
        'final_position_value': info.get('position_value', 0)
    }


def evaluate_agent(env: MarketMakingEnv, agent: TD3Agent, num_episodes: int = 5) -> Dict:
    """Evaluate agent performance"""
    
    agent.set_eval_mode()
    
    eval_rewards = []
    eval_trades = []
    eval_pnls = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action without noise
            action = agent.select_action(state, add_noise=False)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
        
        eval_rewards.append(episode_reward)
        eval_trades.append(info.get('fill_count', 0))
        eval_pnls.append(info.get('realized_pnl', 0) + info.get('unrealized_pnl', 0))
    
    agent.set_train_mode()
    
    return {
        'eval_reward_mean': np.mean(eval_rewards),
        'eval_reward_std': np.std(eval_rewards),
        'eval_trades_mean': np.mean(eval_trades),
        'eval_pnl_mean': np.mean(eval_pnls),
        'eval_pnl_std': np.std(eval_pnls)
    }


def main():
    """Main training function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train TD3 Market Making Agent')
    parser.add_argument('--config', type=str, default='config/td3_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate, do not train')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Starting training with config: {args.config}")
    
    # Setup wandb
    setup_wandb(config)
    
    # Create environment
    env = create_environment(config)
    logger.info(f"Created environment with observation space: {env.observation_space.shape}")
    logger.info(f"Created environment with action space: {env.action_space.shape}")
    
    # Create agent
    agent = create_agent(env, config)
    logger.info("Created TD3 agent")
    
    # Load checkpoint if resuming
    if args.resume:
        agent.load(args.resume)
        logger.info(f"Loaded checkpoint from {args.resume}")
    
    # Setup metrics tracking
    metrics = MetricsTracker()
    
    # Evaluation only mode
    if args.eval_only:
        logger.info("Running evaluation only")
        eval_metrics = evaluate_agent(env, agent, num_episodes=10)
        logger.info(f"Evaluation results: {eval_metrics}")
        return
    
    # Training parameters
    total_episodes = config['training']['total_timesteps'] // config['environment']['max_episode_steps']
    eval_frequency = config['training']['eval_frequency'] // config['environment']['max_episode_steps']
    save_frequency = config['training'].get('save_frequency', 10000) // config['environment']['max_episode_steps']
    
    logger.info(f"Starting training for {total_episodes} episodes")
    
    # Training loop
    best_reward = -float('inf')
    
    for episode in range(total_episodes):
        # Train episode
        episode_metrics = train_episode(env, agent, metrics, episode, logger)
        
        # Log episode metrics
        metrics.log_episode_metrics(episode_metrics, episode)
        
        # Wandb logging
        if config['training']['use_wandb']:
            wandb.log({
                'episode': episode,
                'episode_reward': episode_metrics['episode_reward'],
                'episode_trades': episode_metrics['episode_trades'],
                'episode_pnl': episode_metrics['episode_pnl'],
                'inventory': episode_metrics['final_inventory'],
                **agent.get_stats()
            })
        
        # Evaluation
        if episode % eval_frequency == 0 and episode > 0:
            eval_metrics = evaluate_agent(env, agent)
            logger.info(f"Episode {episode} evaluation: {eval_metrics}")
            
            if config['training']['use_wandb']:
                wandb.log({
                    'episode': episode,
                    **eval_metrics
                })
            
            # Save best model
            if eval_metrics['eval_reward_mean'] > best_reward:
                best_reward = eval_metrics['eval_reward_mean']
                
                # Create checkpoint directory
                checkpoint_dir = config['training']['checkpoint_dir']
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save best model
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                agent.save(best_model_path)
                logger.info(f"Saved best model to {best_model_path}")
        
        # Periodic saving
        if episode % save_frequency == 0 and episode > 0:
            checkpoint_dir = config['training']['checkpoint_dir']
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_episode_{episode}.pth')
            agent.save(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Final evaluation
    logger.info("Training completed. Running final evaluation...")
    final_eval_metrics = evaluate_agent(env, agent, num_episodes=20)
    logger.info(f"Final evaluation results: {final_eval_metrics}")
    
    # Save final model
    model_dir = config['training']['model_save_dir']
    os.makedirs(model_dir, exist_ok=True)
    
    final_model_path = os.path.join(model_dir, 'final_model.pth')
    agent.save(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Close wandb
    if config['training']['use_wandb']:
        wandb.finish()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
