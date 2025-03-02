import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from env import BinPackingEnv
from models import Encoder, CoarsePolicy, FinePolicy, MetaNetwork
from agents import PPOAgent
from utils.logging import Logger, VisualLogger


def generate_items_by_difficulty(
    num_items: int, 
    bin_size: List[int], 
    difficulty: str = 'easy'
) -> List[List[int]]:
    """Generate items based on difficulty level.
    
    Args:
        num_items: Number of items to generate
        bin_size: [W, L, H] dimensions of the bin
        difficulty: 'easy', 'medium', or 'hard'
        
    Returns:
        List of items, each with dimensions [w, l, h]
    """
    W, L, H = bin_size
    items = []
    
    # Determine size ranges based on difficulty
    if difficulty == 'easy':
        # Small items, mostly cubes, low variance
        size_range = (1, max(2, int(min(W, L, H) * 0.25)))
        cuboid_prob = 0.7  # Probability of generating cuboid
    elif difficulty == 'medium':
        # Medium-sized items, mixed shapes, medium variance
        size_range = (1, max(3, int(min(W, L, H) * 0.4)))
        cuboid_prob = 0.5
    else:  # 'hard'
        # Larger items, irregular shapes, high variance
        size_range = (1, max(4, int(min(W, L, H) * 0.6)))
        cuboid_prob = 0.3
    
    for _ in range(num_items):
        # Determine if item is cube-like (similar dimensions) or irregular
        if np.random.random() < cuboid_prob:
            # Create cube-like item (all dimensions similar)
            size = np.random.randint(size_range[0], size_range[1] + 1)
            # Add small variations
            w = max(1, int(size + np.random.randint(-1, 2)))
            l = max(1, int(size + np.random.randint(-1, 2)))
            h = max(1, int(size + np.random.randint(-1, 2)))
        else:
            # Create irregular item (dimensions vary more)
            w = np.random.randint(size_range[0], size_range[1] + 1)
            l = np.random.randint(size_range[0], size_range[1] + 1)
            h = np.random.randint(size_range[0], size_range[1] + 1)
            
        # Ensure items aren't too large for the bin
        w = min(w, W)
        l = min(l, L)
        h = min(h, H)
        
        items.append([w, l, h])
    
    return items


def generate_guillotine_items(
    W: int, 
    L: int, 
    H: int, 
    num_items: int, 
    min_size: int = 1,
    max_size: int = 5
) -> List[List[int]]:
    """Generate items using guillotine cutting.
    
    Args:
        W, L, H: Dimensions of the bin
        num_items: Target number of items to generate
        min_size: Minimum item dimension
        max_size: Maximum item dimension
        
    Returns:
        List of items, each with dimensions [w, l, h]
    """
    items = []
    current_blocks = [[0, 0, 0, W, L, H]]  # [x, y, z, width, length, height]
    
    while len(items) < num_items and current_blocks:
        # Pick a random block to cut
        idx = np.random.randint(0, len(current_blocks))
        x, y, z, w, l, h = current_blocks.pop(idx)
        
        # If block is too small, skip it
        if w < min_size or l < min_size or h < min_size:
            continue
        
        # Choose a random cutting dimension (0=width, 1=length, 2=height)
        cut_dim = np.random.randint(0, 3)
        
        if cut_dim == 0 and w > min_size * 2:  # Cut along width
            cut_pos = np.random.randint(min_size, w - min_size + 1)
            # Create two blocks from the cut
            current_blocks.append([x, y, z, cut_pos, l, h])
            current_blocks.append([x + cut_pos, y, z, w - cut_pos, l, h])
        elif cut_dim == 1 and l > min_size * 2:  # Cut along length
            cut_pos = np.random.randint(min_size, l - min_size + 1)
            # Create two blocks from the cut
            current_blocks.append([x, y, z, w, cut_pos, h])
            current_blocks.append([x, y + cut_pos, z, w, l - cut_pos, h])
        elif cut_dim == 2 and h > min_size * 2:  # Cut along height
            cut_pos = np.random.randint(min_size, h - min_size + 1)
            # Create two blocks from the cut
            current_blocks.append([x, y, z, w, l, cut_pos])
            current_blocks.append([x, y, z + cut_pos, w, l, h - cut_pos])
        else:
            # Block is suitable as an item, add it if within size constraints
            if w <= max_size and l <= max_size and h <= max_size:
                items.append([w, l, h])
            continue
    
    # If we need more items, generate them randomly
    while len(items) < num_items:
        w = np.random.randint(min_size, min(max_size, W) + 1)
        l = np.random.randint(min_size, min(max_size, L) + 1)
        h = np.random.randint(min_size, min(max_size, H) + 1)
        items.append([w, l, h])
    
    return items[:num_items]  # Ensure we return exactly num_items


def get_curriculum_level(
    curriculum_stage: int, 
    max_stages: int = 5,
    bin_size: List[int] = [10, 10, 10]
) -> Dict:
    """Get curriculum parameters for the current stage.
    
    Args:
        curriculum_stage: Current stage of curriculum (0 to max_stages-1)
        max_stages: Maximum number of curriculum stages
        bin_size: [W, L, H] dimensions of the bin
        
    Returns:
        Dictionary with curriculum parameters
    """
    W, L, H = bin_size
    
    # Normalize stage to 0-1 range
    progress = curriculum_stage / (max_stages - 1)
    
    # Scale parameters with curriculum progress
    num_items = int(10 + progress * 40)  # From 10 to 50 items
    
    # Mix of difficulty levels changes with progress
    if progress < 0.2:  # Stage 1: Mostly easy
        difficulties = ['easy'] * 80 + ['medium'] * 20
    elif progress < 0.4:  # Stage 2: More medium
        difficulties = ['easy'] * 60 + ['medium'] * 35 + ['hard'] * 5
    elif progress < 0.6:  # Stage 3: Balanced
        difficulties = ['easy'] * 40 + ['medium'] * 40 + ['hard'] * 20
    elif progress < 0.8:  # Stage 4: More hard
        difficulties = ['easy'] * 20 + ['medium'] * 40 + ['hard'] * 40
    else:  # Stage 5: Mostly hard
        difficulties = ['easy'] * 10 + ['medium'] * 30 + ['hard'] * 60
    
    # Probability of using guillotine cutting increases with difficulty
    use_guillotine_prob = 0.2 + progress * 0.5  # From 0.2 to 0.7
    
    # Size constraints for guillotine items
    min_size = 1
    max_size = int(2 + progress * 4)  # From 2 to 6
    
    return {
        'num_items': num_items,
        'difficulties': difficulties,
        'use_guillotine_prob': use_guillotine_prob,
        'min_size': min_size,
        'max_size': max_size
    }


def generate_curriculum_data(
    num_instances: int,
    bin_size: List[int],
    curriculum_params: Dict
) -> List[List[List[int]]]:
    """Generate data for a curriculum stage.
    
    Args:
        num_instances: Number of problem instances to generate
        bin_size: [W, L, H] dimensions of the bin
        curriculum_params: Parameters from get_curriculum_level
        
    Returns:
        List of item lists for each instance
    """
    all_instances = []
    W, L, H = bin_size
    
    for _ in range(num_instances):
        # Decide whether to use guillotine cutting
        if np.random.random() < curriculum_params['use_guillotine_prob']:
            # Generate using guillotine cutting
            items = generate_guillotine_items(
                W, L, H, 
                curriculum_params['num_items'],
                curriculum_params['min_size'],
                curriculum_params['max_size']
            )
        else:
            # Generate using difficulty-based method
            difficulty = np.random.choice(curriculum_params['difficulties'])
            items = generate_items_by_difficulty(
                curriculum_params['num_items'],
                bin_size,
                difficulty
            )
        
        all_instances.append(items)
    
    return all_instances


def evaluate_agent(
    agent: PPOAgent,
    eval_instances: List[List[List[int]]],
    num_eval_episodes: int = 5
) -> Dict:
    """Evaluate agent on evaluation instances.
    
    Args:
        agent: Trained agent
        eval_instances: List of item lists for evaluation
        num_eval_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    rewards = []
    utilizations = []
    constraint_violations = []
    
    # Select random instances for evaluation
    eval_indices = np.random.choice(len(eval_instances), min(num_eval_episodes, len(eval_instances)), replace=False)
    
    for idx in eval_indices:
        item_list = eval_instances[idx]
        
        # Set item list and reset environment
        agent.env.set_item_list(item_list)
        observation, info = agent.env.reset()
        
        episode_reward = 0
        violations = 0
        steps = 0
        done = False
        
        while not done:
            # Choose action
            action, action_prob, _, debug_info = agent.choose_action(observation)
            
            # Take step in environment
            next_observation, reward, terminated, truncated, info = agent.env.step(action)
            
            # Update metrics
            episode_reward += reward
            if debug_info['constraint_violation'] > 0:
                violations += 1
            steps += 1
            
            # Update observation
            observation = next_observation
            
            # Check termination
            done = terminated or truncated
        
        # Record metrics
        rewards.append(episode_reward)
        utilizations.append(info['volume_utilization_ratio'])
        constraint_violations.append(violations / max(1, steps))
    
    # Calculate averages
    return {
        'avg_reward': np.mean(rewards),
        'avg_utilization': np.mean(utilizations),
        'avg_constraint_violations': np.mean(constraint_violations)
    }


def should_advance_curriculum(
    eval_metrics: Dict,
    target_reward: float,
    target_violations: float
) -> bool:
    """Determine if we should advance to the next curriculum stage.
    
    Args:
        eval_metrics: Metrics from evaluation
        target_reward: Target reward to advance curriculum
        target_violations: Maximum allowed constraint violations
        
    Returns:
        True if we should advance, False otherwise
    """
    # Check if agent has achieved targets
    reward_achieved = eval_metrics['avg_reward'] >= target_reward
    violations_acceptable = eval_metrics['avg_constraint_violations'] <= target_violations
    
    return reward_achieved and violations_acceptable


def plot_training_progress(
    rewards: List[float],
    utilizations: List[float],
    violations: List[float],
    curriculum_stages: List[int],
    save_path: str
):
    """Plot training progress.
    
    Args:
        rewards: List of rewards per episode
        utilizations: List of utilizations per episode
        violations: List of constraint violations per episode
        curriculum_stages: List of curriculum stages per episode
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot rewards
    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Mark curriculum stage changes
    stage_changes = [i for i in range(1, len(curriculum_stages)) if curriculum_stages[i] != curriculum_stages[i-1]]
    for stage in stage_changes:
        plt.axvline(x=stage, color='r', linestyle='--', alpha=0.5)
    
    # Plot utilizations
    plt.subplot(3, 1, 2)
    plt.plot(utilizations)
    plt.title('Volume Utilization')
    plt.xlabel('Episode')
    plt.ylabel('Utilization')
    
    # Mark curriculum stage changes
    for stage in stage_changes:
        plt.axvline(x=stage, color='r', linestyle='--', alpha=0.5)
    
    # Plot violations
    plt.subplot(3, 1, 3)
    plt.plot(violations)
    plt.title('Constraint Violations')
    plt.xlabel('Episode')
    plt.ylabel('Violation Rate')
    
    # Mark curriculum stage changes
    for stage in stage_changes:
        plt.axvline(x=stage, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_with_curriculum(
    bin_size: List[int] = [10, 10, 10],
    buffer_size: int = 5,
    num_episodes: int = 5000,
    max_curriculum_stages: int = 5,
    episodes_per_stage: int = 1000,
    eval_interval: int = 50,
    save_path: str = './saved_models',
    log_dir: str = './logs',
    device: Optional[torch.device] = None
):
    """Train the agent using curriculum learning.
    
    Args:
        bin_size: [W, L, H] dimensions of the bin
        buffer_size: Number of items in buffer
        num_episodes: Maximum number of episodes to train
        max_curriculum_stages: Number of curriculum stages
        episodes_per_stage: Maximum episodes per curriculum stage
        eval_interval: Interval for evaluation
        save_path: Path to save models
        log_dir: Directory for logs
        device: Device to run on
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on device: {device}")
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize logger
    experiment_name = f"bin_packing_{int(time.time())}"
    logger = Logger(log_dir=log_dir, experiment_name=experiment_name, use_tensorboard=True)
    
    # Initialize environment with empty item list (will be updated in training loop)
    env = BinPackingEnv(bin_size=bin_size, buffer_size=buffer_size, item_list=[])
    
    # Initialize models
    encoder = Encoder(
        W=bin_size[0], 
        L=bin_size[1], 
        H=bin_size[2], 
        buffer_size=buffer_size, 
        embedding_dim=256
    ).to(device)
    
    coarse_policy = CoarsePolicy(
        W=bin_size[0], 
        L=bin_size[1], 
        buffer_size=buffer_size, 
        embedding_dim=256, 
        coarse_grid_size=(5, 5)
    ).to(device)
    
    fine_policy = FinePolicy(
        W=bin_size[0], 
        L=bin_size[1], 
        H=bin_size[2], 
        embedding_dim=256
    ).to(device)
    
    meta_network = MetaNetwork(
        W=bin_size[0], 
        L=bin_size[1], 
        embedding_dim=256
    ).to(device)
    
    # Initialize agent
    agent = PPOAgent(
        env=env,
        encoder=encoder,
        coarse_policy=coarse_policy,
        fine_policy=fine_policy,
        meta_network=meta_network,
        coarse_grid_size=(5, 5),
        device=device,
        logger=logger
    )
    
    # Generate validation data for consistent evaluation
    eval_instances = []
    for stage in range(max_curriculum_stages):
        curriculum_params = get_curriculum_level(stage, max_curriculum_stages, bin_size)
        stage_instances = generate_curriculum_data(5, bin_size, curriculum_params)
        eval_instances.extend(stage_instances)
    
    # Training variables
    current_stage = 0
    total_episodes = 0
    best_eval_reward = float('-inf')
    
    # Metrics tracking
    rewards_history = []
    utilizations_history = []
    violations_history = []
    curriculum_stages_history = []
    
    # Start training
    print(f"Starting curriculum learning with {max_curriculum_stages} stages")
    
    while total_episodes < num_episodes and current_stage < max_curriculum_stages:
        # Get curriculum parameters for current stage
        curriculum_params = get_curriculum_level(current_stage, max_curriculum_stages, bin_size)
        
        print(f"\nCurriculum Stage {current_stage+1}/{max_curriculum_stages}")
        print(f"Parameters: {curriculum_params}")
        
        # Generate training data for this stage
        train_instances = generate_curriculum_data(50, bin_size, curriculum_params)
        
        # Train for this stage
        stage_episodes = 0
        
        while stage_episodes < episodes_per_stage and total_episodes < num_episodes:
            # Select a random instance
            instance_idx = np.random.randint(0, len(train_instances))
            item_list = train_instances[instance_idx]
            
            # Set item list and reset environment
            env.set_item_list(item_list)
            observation, info = env.reset()
            
            # Run episode
            episode_reward = 0
            violations = 0
            steps = 0
            done = False
            
            while not done:
                # Choose action
                action, action_prob, state_embed, debug_info = agent.choose_action(observation)
                
                # Take step in environment
                next_observation, reward, terminated, truncated, info = env.step(action)
                
                # Store transition in memory
                agent.memory.store(
                    observation=observation,
                    state=state_embed,
                    action=action,
                    prob=action_prob,
                    val=agent.value_network(state_embed).item(),
                    reward=reward,
                    done=terminated or truncated
                )
                
                # Update metrics
                episode_reward += reward
                if debug_info['constraint_violation'] > 0:
                    violations += 1
                steps += 1
                
                # Update observation
                observation = next_observation
                
                # Check termination
                done = terminated or truncated
            
            # Log episode metrics
            utilization = info['volume_utilization_ratio']
            violation_rate = violations / max(1, steps)
            
            logger.log_scalar('episode_reward', episode_reward, total_episodes)
            logger.log_scalar('utilization', utilization, total_episodes)
            logger.log_scalar('constraint_violations', violation_rate, total_episodes)
            logger.log_scalar('curriculum_stage', current_stage, total_episodes)
            
            # Store for plotting
            rewards_history.append(episode_reward)
            utilizations_history.append(utilization)
            violations_history.append(violation_rate)
            curriculum_stages_history.append(current_stage)
            
            # Train networks if memory is full
            if agent.memory.is_full():
                metrics = agent.train_networks()
                for key, value in metrics.items():
                    logger.log_scalar(f'train/{key}', value, total_episodes)
            
            # Increment counters
            stage_episodes += 1
            total_episodes += 1
            
            # Print progress
            if total_episodes % 10 == 0:
                print(f"Episode {total_episodes}/{num_episodes} | " 
                      f"Stage {current_stage+1}/{max_curriculum_stages} | "
                      f"Reward: {episode_reward:.4f} | "
                      f"Utilization: {utilization:.2%} | "
                      f"Violations: {violation_rate:.2%}")
            
            # Evaluate agent
            if total_episodes % eval_interval == 0:
                eval_metrics = evaluate_agent(agent, eval_instances, num_eval_episodes=10)
                
                print(f"\nEvaluation | "
                      f"Reward: {eval_metrics['avg_reward']:.4f} | "
                      f"Utilization: {eval_metrics['avg_utilization']:.2%} | "
                      f"Violations: {eval_metrics['avg_constraint_violations']:.2%}")
                
                # Log evaluation metrics
                for key, value in eval_metrics.items():
                    logger.log_scalar(f'eval/{key}', value, total_episodes)
                
                # Save best model
                if eval_metrics['avg_reward'] > best_eval_reward:
                    best_eval_reward = eval_metrics['avg_reward']
                    agent.save_models(os.path.join(save_path, 'best_model'))
                    print(f"New best model saved with reward {best_eval_reward:.4f}")
                
                # Check if we should advance to next curriculum stage
                target_reward = 0.4 + 0.1 * current_stage  # Increasing targets with stage
                target_violations = 0.3 - 0.05 * current_stage  # Decreasing allowed violations
                
                if should_advance_curriculum(eval_metrics, target_reward, target_violations):
                    print(f"Advancing to next curriculum stage!")
                    current_stage += 1
                    break
        
        # Force curriculum advancement if max episodes for stage reached
        if stage_episodes >= episodes_per_stage:
            print(f"Maximum episodes for stage reached. Advancing curriculum.")
            current_stage += 1
    
    # Training complete
    print("\nTraining complete!")
    
    # Save final model
    agent.save_models(os.path.join(save_path, 'final_model'))
    print(f"Final model saved to {os.path.join(save_path, 'final_model')}")
    
    # Plot training progress
    plot_path = os.path.join(log_dir, experiment_name, 'training_progress.png')
    plot_training_progress(
        rewards_history,
        utilizations_history,
        violations_history,
        curriculum_stages_history,
        plot_path
    )
    print(f"Training progress plot saved to {plot_path}")
    
    # Close logger
    logger.close()


if __name__ == "__main__":
    # Training parameters
    bin_size = [10, 10, 10]
    buffer_size = 5
    num_episodes = 5000
    
    # Start training
    train_with_curriculum(
        bin_size=bin_size,
        buffer_size=buffer_size,
        num_episodes=num_episodes,
        max_curriculum_stages=5,
        episodes_per_stage=1000,
        eval_interval=50,
        save_path='./saved_models',
        log_dir='./logs'
    )