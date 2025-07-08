"""
Evaluation Script for TD3 Market Making Agent

This script evaluates a trained TD3 agent and generates performance reports.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from typing import Dict, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.market_env import MarketMakingEnv
from agent.td3_agent import TD3Agent
from utils.logger import setup_logger
from utils.metrics import MetricsTracker, PerformanceAnalyzer


def create_environment(config: Dict) -> MarketMakingEnv:
    """Create evaluation environment"""
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
    
    env = MarketMakingEnv(
        symbol=config['data']['symbols'][0],
        config=env_config,
        data_source="synthetic"
    )
    
    return env


def load_agent(env: MarketMakingEnv, config: Dict, model_path: str) -> TD3Agent:
    """Load trained agent"""
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
    
    agent.load(model_path)
    agent.set_eval_mode()
    
    return agent


def evaluate_episode(env: MarketMakingEnv, agent: TD3Agent, episode_num: int, 
                    render: bool = False) -> Dict:
    """Evaluate single episode"""
    
    state = env.reset()
    
    episode_reward = 0
    episode_steps = 0
    episode_trades = 0
    episode_pnl = 0
    inventory_history = []
    pnl_history = []
    action_history = []
    
    done = False
    
    while not done:
        # Select action (no exploration noise)
        action = agent.select_action(state, add_noise=False)
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        # Record metrics
        episode_reward += reward
        episode_steps += 1
        episode_trades = info.get('fill_count', 0)
        episode_pnl = info.get('realized_pnl', 0) + info.get('unrealized_pnl', 0)
        
        # Store history
        inventory_history.append(info.get('inventory', 0))
        pnl_history.append(episode_pnl)
        action_history.append(action.copy())
        
        state = next_state
        
        # Optional rendering
        if render and episode_steps % 1000 == 0:
            env.render()
    
    return {
        'episode': episode_num,
        'episode_reward': episode_reward,
        'episode_steps': episode_steps,
        'episode_trades': episode_trades,
        'episode_pnl': episode_pnl,
        'final_inventory': info.get('inventory', 0),
        'final_position_value': info.get('position_value', 0),
        'inventory_history': inventory_history,
        'pnl_history': pnl_history,
        'action_history': action_history,
        'info': info
    }


def run_evaluation(env: MarketMakingEnv, agent: TD3Agent, num_episodes: int = 10, 
                  render: bool = False) -> List[Dict]:
    """Run full evaluation"""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting evaluation with {num_episodes} episodes")
    
    results = []
    
    for episode in range(num_episodes):
        logger.info(f"Running episode {episode + 1}/{num_episodes}")
        
        result = evaluate_episode(env, agent, episode, render)
        results.append(result)
        
        logger.info(f"Episode {episode + 1} completed: "
                   f"Reward={result['episode_reward']:.2f}, "
                   f"PnL={result['episode_pnl']:.2f}, "
                   f"Trades={result['episode_trades']}, "
                   f"Inventory={result['final_inventory']}")
    
    return results


def analyze_results(results: List[Dict], save_dir: str = 'evaluation_results'):
    """Analyze and save evaluation results"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract metrics
    rewards = [r['episode_reward'] for r in results]
    pnls = [r['episode_pnl'] for r in results]
    trades = [r['episode_trades'] for r in results]
    inventories = [r['final_inventory'] for r in results]
    
    # Calculate statistics
    stats = {
        'num_episodes': len(results),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_pnl': np.mean(pnls),
        'std_pnl': np.std(pnls),
        'mean_trades': np.mean(trades),
        'std_trades': np.std(trades),
        'mean_inventory': np.mean(np.abs(inventories)),
        'max_inventory': np.max(np.abs(inventories)),
        'positive_pnl_episodes': sum(1 for pnl in pnls if pnl > 0),
        'hit_ratio': sum(1 for pnl in pnls if pnl > 0) / len(pnls),
        'total_pnl': sum(pnls),
        'sharpe_ratio': np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
    }
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Episodes: {stats['num_episodes']}")
    print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Mean PnL: {stats['mean_pnl']:.2f} ± {stats['std_pnl']:.2f}")
    print(f"Total PnL: {stats['total_pnl']:.2f}")
    print(f"Mean Trades: {stats['mean_trades']:.2f} ± {stats['std_trades']:.2f}")
    print(f"Mean Inventory: {stats['mean_inventory']:.2f}")
    print(f"Max Inventory: {stats['max_inventory']:.2f}")
    print(f"Hit Ratio: {stats['hit_ratio']:.2f}")
    print(f"Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
    print("="*50)
    
    # Save statistics
    with open(os.path.join(save_dir, 'evaluation_stats.txt'), 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Episode rewards
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Episode PnL
    axes[0, 1].plot(pnls)
    axes[0, 1].set_title('Episode PnL')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('PnL')
    axes[0, 1].grid(True)
    
    # Cumulative PnL
    cumulative_pnl = np.cumsum(pnls)
    axes[0, 2].plot(cumulative_pnl)
    axes[0, 2].set_title('Cumulative PnL')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Cumulative PnL')
    axes[0, 2].grid(True)
    
    # Episode trades
    axes[1, 0].plot(trades)
    axes[1, 0].set_title('Episode Trades')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Number of Trades')
    axes[1, 0].grid(True)
    
    # Final inventories
    axes[1, 1].plot(inventories)
    axes[1, 1].set_title('Final Inventories')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Inventory')
    axes[1, 1].grid(True)
    
    # PnL histogram
    axes[1, 2].hist(pnls, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('PnL Distribution')
    axes[1, 2].set_xlabel('PnL')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot detailed episode analysis for first few episodes
    if len(results) > 0:
        plot_episode_details(results[:min(3, len(results))], save_dir)
    
    return stats


def plot_episode_details(results: List[Dict], save_dir: str):
    """Plot detailed analysis for specific episodes"""
    
    fig, axes = plt.subplots(len(results), 3, figsize=(15, 5 * len(results)))
    
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        episode = result['episode']
        
        # Inventory over time
        axes[i, 0].plot(result['inventory_history'])
        axes[i, 0].set_title(f'Episode {episode}: Inventory Over Time')
        axes[i, 0].set_xlabel('Step')
        axes[i, 0].set_ylabel('Inventory')
        axes[i, 0].grid(True)
        
        # PnL over time
        axes[i, 1].plot(result['pnl_history'])
        axes[i, 1].set_title(f'Episode {episode}: PnL Over Time')
        axes[i, 1].set_xlabel('Step')
        axes[i, 1].set_ylabel('PnL')
        axes[i, 1].grid(True)
        
        # Actions over time
        actions = np.array(result['action_history'])
        if len(actions) > 0:
            axes[i, 2].plot(actions[:, 0], label='Bid Offset', alpha=0.7)
            axes[i, 2].plot(actions[:, 1], label='Ask Offset', alpha=0.7)
            axes[i, 2].set_title(f'Episode {episode}: Actions Over Time')
            axes[i, 2].set_xlabel('Step')
            axes[i, 2].set_ylabel('Action Value')
            axes[i, 2].legend()
            axes[i, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'episode_details.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main evaluation function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate TD3 Market Making Agent')
    parser.add_argument('--config', type=str, default='config/td3_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during evaluation')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logger("evaluation", level="INFO")
    logger.info(f"Starting evaluation with model: {args.model}")
    
    # Create environment
    env = create_environment(config)
    logger.info("Created evaluation environment")
    
    # Load agent
    agent = load_agent(env, config, args.model)
    logger.info("Loaded trained agent")
    
    # Run evaluation
    results = run_evaluation(env, agent, args.episodes, args.render)
    
    # Analyze results
    stats = analyze_results(results, args.save_dir)
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
