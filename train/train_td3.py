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
        global wandb
        import wandb
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
        'spread_penalty': config['environment']['reward_config']['spread_penalty'],
        'historical_data_path': config['data'].get('historical_data_path'),
        'quote_size': config['environment'].get('quote_size', 100),
    }
    
    # Create environment
    env = MarketMakingEnv(
        symbol=config['data']['symbols'][0],  # Use first symbol for training
        config=env_config,
        data_source=config['data'].get('source', 'historical')
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
                 episode: int, logger: logging.Logger, step_limit: int = None) -> Dict:
    """Train one episode, optionally capped by the remaining invocation budget.

    A budget cutoff is a truncation, not an environment terminal transition:
    retain the environment's done flag in replay for TD3 bootstrapping.
    """
    
    # Reset environment
    state = env.reset()
    
    episode_reward = 0
    episode_steps = 0
    episode_trades = 0
    episode_pnl = 0
    
    done = False
    
    while not done and (step_limit is None or episode_steps < step_limit):
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
        'final_position_value': info.get('position_value', 0),
        'budget_truncated': not done,
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


def training_schedule(config: Dict) -> Dict:
    """Validate positive action-step budgets and intervals before side effects.

    training.save_frequency overrides the legacy agent.save_frequency setting.
    Frequencies count actual training transitions, not presumed episode length.
    """
    training = config['training']
    schedule = {
        'total_timesteps': training['total_timesteps'],
        'eval_frequency': training['eval_frequency'],
        'save_frequency': training.get('save_frequency', config['agent'].get('save_frequency', 10000)),
        'max_episode_steps': config['environment']['max_episode_steps'],
    }
    for name, value in schedule.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer number of action steps')
    return schedule


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
    try:
        schedule = training_schedule(config)
    except ValueError as error:
        parser.error(str(error))
    
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
    total_steps = 0
    episode = 0
    next_eval = schedule['eval_frequency']
    next_save = schedule['save_frequency']
    logger.info(f"Starting training for {schedule['total_timesteps']} action steps")
    
    # Training loop
    best_reward = -float('inf')
    
    while total_steps < schedule['total_timesteps']:
        # Train episode
        episode_metrics = train_episode(env, agent, metrics, episode, logger,
                                        step_limit=schedule['total_timesteps'] - total_steps)
        total_steps += episode_metrics['episode_steps']
        
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
        # Run once at an episode boundary that crosses a step threshold. Avoid
        # duplicate evaluations/checkpoints when a long episode spans intervals.
        if total_steps >= next_eval:
            next_eval = (total_steps // schedule['eval_frequency'] + 1) * schedule['eval_frequency']
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
        if total_steps >= next_save:
            next_save = (total_steps // schedule['save_frequency'] + 1) * schedule['save_frequency']
            checkpoint_dir = config['training']['checkpoint_dir']
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_steps_{total_steps}.pth')
            agent.save(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        episode += 1
    
    # Final evaluation
    logger.info(f"Training completed after {total_steps} action steps across {episode} episodes. Running final evaluation...")
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

