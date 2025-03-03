import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, List, Optional, Union, Any, Callable
import time
import os
from collections import deque

from models import Encoder, CoarsePolicy, FinePolicy, MetaNetwork
from .value_network import ValueNetwork, AdvantageEstimator
from utils.masks import create_coarse_mask, create_fine_mask
from utils.gradients import compute_objective_function, compute_support_ratio
from utils.heuristics import heuristic_refinement
from utils.logging import Logger


class PPOMemory:
    """Memory buffer for PPO algorithm to store trajectories."""
    
    def __init__(
        self, 
        batch_size: int,
        mini_batch_size: int,
        state_dim: int,
        action_dim: int = 4  # (b, r, x, y)
    ):
        """Initialize the PPO memory buffer.
        
        Args:
            batch_size: Number of transitions to collect before update
            mini_batch_size: Size of mini-batches for training
            state_dim: Dimension of encoded state
            action_dim: Dimension of action space (default: 4 for bin packing)
        """
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize memory buffers
        self.clear()
    
    def clear(self):
        """Clear memory buffers."""
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.observations = []  # Raw observations
        
        self.batch_idx = 0
    
    def store(
        self,
        observation: Dict[str, np.ndarray],
        state: torch.Tensor,
        action: np.ndarray,
        prob: float,
        val: float,
        reward: float,
        done: bool
    ):
        """Store a transition in memory.
        
        Args:
            observation: Raw observation from environment
            state: Encoded state vector
            action: Action taken
            prob: Action probability
            val: State value
            reward: Reward received
            done: Whether episode terminated
        """
        self.observations.append(observation)
        self.states.append(state.cpu().detach().numpy())
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
        
        self.batch_idx += 1
    
    def is_full(self) -> bool:
        """Check if memory is full and ready for training."""
        return self.batch_idx >= self.batch_size
    
    def _get_tensor(self, data: List, device: torch.device) -> torch.Tensor:
        """Convert list to tensor and move to device."""
        return torch.tensor(np.array(data), dtype=torch.float32).to(device)
    
    def generate_batches(self, device: torch.device) -> List[Dict[str, torch.Tensor]]:
        """Generate mini-batches for training.
        
        Args:
            device: Device to place tensors on
            
        Returns:
            List of dictionaries containing mini-batches
        """
        n_states = len(self.states)
        batch_start_indices = np.arange(0, n_states, self.mini_batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = []
        
        for start_idx in batch_start_indices:
            end_idx = min(start_idx + self.mini_batch_size, n_states)
            mini_batch_indices = indices[start_idx:end_idx]
            
            mini_batch = {
                'states': self._get_tensor([self.states[idx] for idx in mini_batch_indices], device),
                'actions': self._get_tensor([self.actions[idx] for idx in mini_batch_indices], device),
                'old_probs': self._get_tensor([self.probs[idx] for idx in mini_batch_indices], device),
                'vals': self._get_tensor([self.vals[idx] for idx in mini_batch_indices], device),
                'rewards': self._get_tensor([self.rewards[idx] for idx in mini_batch_indices], device),
                'dones': self._get_tensor([self.dones[idx] for idx in mini_batch_indices], device),
                'observations': [self.observations[idx] for idx in mini_batch_indices]
            }
            
            batches.append(mini_batch)
        
        return batches


class DynamicHyperparameters:
    """Handler for dynamic hyperparameters adjustment with adaptive decay."""
    
    def __init__(
        self,
        alpha_init: float = 0.7,       # Điều chỉnh giá trị khởi tạo
        beta_init: float = 0.3,        # Điều chỉnh giá trị khởi tạo
        lambda_init: float = 0.3,
        base_decay_factor: float = 0.995,
        min_value: float = 0.1,        # Tăng giá trị tối thiểu
        max_value: float = 0.9,
        reward_threshold: float = 0.02, # Ngưỡng phát hiện xu hướng reward
        violation_threshold: float = 0.01, # Ngưỡng phát hiện xu hướng vi phạm
        adaptive_factor: float = 0.2   # Hệ số điều chỉnh adaptive
    ):
        """Initialize dynamic hyperparameters.
        
        Args:
            alpha_init: Initial value for volume utilization weight
            beta_init: Initial value for support constraint weight
            lambda_init: Initial value for constraint loss weight
            base_decay_factor: Base decay factor for parameter adjustment
            min_value: Minimum value for parameters
            max_value: Maximum value for parameters
            reward_threshold: Threshold for detecting reward trends
            violation_threshold: Threshold for detecting violation trends
            adaptive_factor: Factor for adapting decay rate based on trend magnitudes
        """
        self.alpha = alpha_init
        self.beta = beta_init
        self.lambda_val = lambda_init
        self.base_decay_factor = base_decay_factor
        self.min_value = min_value
        self.max_value = max_value
        self.reward_threshold = reward_threshold
        self.violation_threshold = violation_threshold
        self.adaptive_factor = adaptive_factor
        
        # Performance tracking
        self.reward_history = deque(maxlen=100)
        self.constraint_violation_history = deque(maxlen=100)
        
        # Tracking episode count
        self.episode_count = 0
        
        # Lambda schedule để tăng dần ảnh hưởng của constraint loss
        self.lambda_schedule = {
            100: 0.4,
            300: 0.5,
            600: 0.6,
            1000: 0.7
        }
    
    def update(self, avg_reward: float, constraint_violations: float):
        """Update hyperparameters based on performance with adaptive decay.
        
        Args:
            avg_reward: Average reward in recent episodes
            constraint_violations: Average constraint violations in recent episodes
        """
        # Tăng episode counter
        self.episode_count += 1
        
        # Kiểm tra schedule lambda
        if self.episode_count in self.lambda_schedule:
            self.lambda_val = self.lambda_schedule[self.episode_count]
        
        # Store metrics
        self.reward_history.append(avg_reward)
        self.constraint_violation_history.append(constraint_violations)
        
        # Skip update if not enough history
        if len(self.reward_history) < 20:  # Tăng lên để có đánh giá xu hướng tốt hơn
            return
        
        # Calculate trends using kỹ thuật moving average
        recent_rewards = list(self.reward_history)[-20:]
        older_rewards = list(self.reward_history)[-40:-20]
        reward_trend = np.mean(recent_rewards) - np.mean(older_rewards)
        
        recent_violations = list(self.constraint_violation_history)[-20:]
        older_violations = list(self.constraint_violation_history)[-40:-20]
        violation_trend = np.mean(recent_violations) - np.mean(older_violations)
        
        # Tính toán adaptive decay dựa trên độ lớn của xu hướng
        reward_magnitude = min(1.0, abs(reward_trend) / 0.1)  # Chuẩn hóa magnitude
        violation_magnitude = min(1.0, abs(violation_trend) / 0.05)  # Chuẩn hóa magnitude
        
        # Decay factor thích ứng
        alpha_decay = self.base_decay_factor ** (1 + self.adaptive_factor * reward_magnitude)
        beta_decay = self.base_decay_factor ** (1 + self.adaptive_factor * violation_magnitude)
        
        # Adjust alpha and beta based on performance
        if violation_trend > self.violation_threshold:  # Constraint violations increasing
            # Decrease alpha (care less about volume) and increase beta (care more about constraints)
            self.alpha *= alpha_decay
            self.beta /= beta_decay
            # Tăng lambda để phạt vi phạm nặng hơn
            self.lambda_val = min(self.max_value, self.lambda_val * 1.1)
        elif reward_trend < -self.reward_threshold:  # Rewards decreasing
            # Increase alpha (care more about volume) and decrease beta slightly
            self.alpha /= alpha_decay
            self.beta *= beta_decay**0.7  # Decrease beta more slowly
        else:
            # Thêm logic điều chỉnh cho trường hợp cả hai xu hướng tốt
            if reward_trend > self.reward_threshold and violation_trend < -self.violation_threshold:
                # Cả hai đều đang tốt, giữ nguyên giá trị
                pass
            else:
                # Điều chỉnh nhẹ cho thăm dò
                self.alpha *= self.base_decay_factor**0.05
                self.beta *= self.base_decay_factor**0.05
        
        # Enforce bounds
        self.alpha = np.clip(self.alpha, self.min_value, self.max_value)
        self.beta = np.clip(self.beta, self.min_value, self.max_value)
        self.lambda_val = np.clip(self.lambda_val, self.min_value, self.max_value)
        
        # Normalize to ensure alpha + beta = 1
        total = self.alpha + self.beta
        self.alpha /= total
        self.beta /= total
    
    def get_values(self) -> Dict[str, float]:
        """Get current hyperparameter values.
        
        Returns:
            Dictionary of current values
        """
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'lambda': self.lambda_val
        }


class PPOAgent:
    """Proximal Policy Optimization agent for 3D Bin Packing problem."""
    
    def __init__(
        self,
        env: gym.Env,
        encoder: Encoder,
        coarse_policy: CoarsePolicy,
        fine_policy: FinePolicy,
        meta_network: MetaNetwork,
        coarse_grid_size: Tuple[int, int] = (5, 5),
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.1,
        value_clip: float = 0.2,
        n_epochs: int = 4,
        batch_size: int = 256,
        mini_batch_size: int = 64,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.3,
        target_kl: float = 0.03,
        use_dynamic_hyperparams: bool = True,
        logger: Optional[Logger] = None
    ):
        """Initialize the PPO agent.
        
        Args:
            env: Gymnasium environment for 3D bin packing
            encoder: Encoder network for state representation
            coarse_policy: Coarse policy network
            fine_policy: Fine policy network
            meta_network: Meta network for ensemble
            coarse_grid_size: Size of the coarse grid
            device: Device to run networks on
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            policy_clip: PPO clipping parameter
            value_clip: Value function clipping parameter
            n_epochs: Number of epochs to train per batch
            batch_size: Number of steps to collect before update
            mini_batch_size: Mini-batch size for training
            entropy_coef: Entropy coefficient for encouraging exploration
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            target_kl: Target KL divergence for early stopping
            use_dynamic_hyperparams: Whether to use dynamic hyperparameters
            logger: Logger for tracking metrics
        """
        self.env = env
        self.encoder = encoder
        self.coarse_policy = coarse_policy
        self.fine_policy = fine_policy
        self.meta_network = meta_network
        
        # Get bin dimensions from environment
        self.bin_size = env.bin_size
        self.W, self.L, self.H = self.bin_size
        self.coarse_grid_size = coarse_grid_size
        
        # Create value network
        embedding_dim = encoder.layer_norm.normalized_shape[0]  # Get embedding dimension from encoder
        self.value_network = ValueNetwork(embedding_dim=embedding_dim).to(device)
        
        # Set up advantage estimator
        self.advantage_estimator = AdvantageEstimator(gamma=gamma, gae_lambda=gae_lambda)
        
        # Set up optimizers
        self.encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
        self.coarse_policy_optimizer = optim.Adam(coarse_policy.parameters(), lr=lr)
        self.fine_policy_optimizer = optim.Adam(fine_policy.parameters(), lr=lr)
        self.meta_network_optimizer = optim.Adam(meta_network.parameters(), lr=lr)
        self.value_network_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)
        
        # Set up memory
        self.memory = PPOMemory(batch_size, mini_batch_size, embedding_dim, action_dim=4)
        
        # Set up hyperparameters
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.value_clip = value_clip
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
        # Set up dynamic hyperparameters
        self.use_dynamic_hyperparams = use_dynamic_hyperparams
        self.dynamic_hyperparams = DynamicHyperparameters() if use_dynamic_hyperparams else None
        
        # Track metrics
        self.logger = logger
        self.episode_rewards = []
        self.constraint_violations = []
        self.episode_counter = 0
        self.step_counter = 0
        
        # Utilities
        self.device = device
    
    def choose_action(
        self, 
        observation: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, float, torch.Tensor, Dict[str, Any]]:
        """Choose an action using the policy networks.
        
        Args:
            observation: Environment observation
            
        Returns:
            Tuple of (action, action_prob, state_embedding, debug_info)
        """
        # Convert observation to tensors
        height_map = torch.tensor(observation['height_map'], dtype=torch.float32).unsqueeze(0).to(self.device)
        buffer = torch.tensor(observation['buffer'], dtype=torch.float32).unsqueeze(0).to(self.device)
        occupancy_map = torch.tensor(observation['occupancy_map'], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        tensor_observation = {
            'height_map': height_map,
            'buffer': buffer,
            'occupancy_map': occupancy_map
        }
        
        # Get hyperparameters for refinement
        hyperparams = self.dynamic_hyperparams.get_values() if self.use_dynamic_hyperparams else {
            'alpha': 0.7, 'beta': 0.3, 'lambda': 0.1
        }
        
        with torch.no_grad():
            # Encode state
            z_t = self.encoder(tensor_observation)
            
            # Create coarse mask
            coarse_mask = create_coarse_mask(
                height_map, buffer, self.bin_size, self.coarse_grid_size
            )
            
            # Get coarse policy logits and probabilities
            coarse_logits, coarse_probs = self.coarse_policy(z_t, coarse_mask)
            
            # Get candidate region
            batch_indices, item_indices, rotations, coarse_x, coarse_y = \
                self.coarse_policy.get_candidate_region(coarse_probs)
            
            # Get region boundaries
            x_min, y_min, x_max, y_max = self.coarse_policy.get_region_boundaries(
                coarse_x, coarse_y
            )
            
            # Get coarse features for skip connections
            coarse_features = self.coarse_policy.get_feature_maps()
            
            # Create region data for fine policy
            region_data = {
                'item_indices': item_indices,
                'rotation': rotations,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'items': torch.stack([buffer[i, idx] for i, idx in enumerate(item_indices)])
            }
            
            # Create fine mask
            fine_mask = create_fine_mask(
                height_map,
                region_data['items'],
                region_data['rotation'],
                self.bin_size,
                (x_min, y_min, x_max, y_max)
            )
            
            # Get fine policy logits and probabilities
            fine_logits, fine_probs = self.fine_policy(
                z_t, region_data, coarse_features, None, fine_mask
            )
            
            # Apply heuristic refinement
            heur_logits, heur_probs = heuristic_refinement(
                height_map, fine_logits, region_data, fine_mask, self.bin_size,
                method='greedy', 
                alpha_t=hyperparams['alpha'],
                beta_t=hyperparams['beta'],
                gamma=0.2
            )
            
            # Ensemble results using meta network
            final_logits, final_probs = self.meta_network(
                fine_logits, heur_logits, fine_mask, z_t
            )
            
            # Select final action
            batch_size = final_probs.size(0)
            final_probs_flat = final_probs.view(batch_size, -1)
            final_indices = torch.argmax(final_probs_flat, dim=1)
            
            # Convert flat indices to coordinates
            x = (final_indices // self.L).long()
            y = (final_indices % self.L).long()
            
            # Create final action
            action = np.array([
                item_indices[0].item(),
                rotations[0].item(),
                x[0].item(),
                y[0].item()
            ])
            
            # Get action probability
            action_prob = final_probs_flat[0, final_indices[0]].item()
            
            # Get state value
            state_value = self.value_network(z_t).squeeze()
            
            # Compute support ratio for tracking constraint violations
            position = torch.tensor([[x[0], y[0]]], device=self.device)
            support_ratio = compute_support_ratio(
                height_map,
                region_data['items'],
                region_data['rotation'],
                position
            )[0].item()
            
            # Debug info
            debug_info = {
                'coarse_position': (coarse_x[0].item(), coarse_y[0].item()),
                'fine_position': (x[0].item(), y[0].item()),
                'item_index': item_indices[0].item(),
                'rotation': rotations[0].item(),
                'support_ratio': support_ratio,
                'constraint_violation': 1.0 if support_ratio < 0.8 and support_ratio > 0 else 0.0,
                'hyperparams': hyperparams
            }
            
        return action, action_prob, z_t, debug_info
    
    def train_networks(self) -> Dict[str, float]:
        """Train policy and value networks using collected experience with improved gradient clipping.
        
        Returns:
            Dictionary of training metrics
        """
        # Generate mini-batches
        batches = self.memory.generate_batches(self.device)
        
        # Track metrics
        metrics = {
            'actor_loss': 0,
            'critic_loss': 0,
            'entropy': 0,
            'kl_div': 0,
            'constraint_loss': 0,
            'total_loss': 0,
            'grad_norm': 0
        }
        
        # Get hyperparameters
        hyperparams = self.dynamic_hyperparams.get_values() if self.use_dynamic_hyperparams else {
            'alpha': 0.7, 'beta': 0.3, 'lambda': 0.1
        }
        
        # Thêm kiểm soát learning rate điều chỉnh tự động
        current_lr = 0
        for param_group in self.encoder_optimizer.param_groups:
            current_lr = param_group['lr']
        
        # Training loop
        for epoch in range(self.n_epochs):
            # Track epoch metrics
            epoch_metrics = {key: 0 for key in metrics.keys()}
            
            # Giảm learning rate ở các epoch đầu tiên để tránh KL cao
            if self.step_counter < 1000:
                lr_scale = min(1.0, self.step_counter / 1000)
                for optimizer in [self.encoder_optimizer, self.coarse_policy_optimizer,
                                self.fine_policy_optimizer, self.meta_network_optimizer,
                                self.value_network_optimizer]:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr * lr_scale
            
            for batch in batches:
                # Get batch data
                states = batch['states']
                actions = batch['actions'].long()
                old_probs = batch['old_probs']
                values = batch['vals']
                rewards = batch['rewards']
                dones = batch['dones']
                observations = batch['observations']
                
                # Zero gradients
                self.encoder_optimizer.zero_grad()
                self.coarse_policy_optimizer.zero_grad()
                self.fine_policy_optimizer.zero_grad()
                self.meta_network_optimizer.zero_grad()
                self.value_network_optimizer.zero_grad()
                
                # Convert observations to tensors
                tensor_observations = []
                for obs in observations:
                    tensor_obs = {
                        'height_map': torch.tensor(obs['height_map'], dtype=torch.float32).unsqueeze(0).to(self.device),
                        'buffer': torch.tensor(obs['buffer'], dtype=torch.float32).unsqueeze(0).to(self.device),
                        'occupancy_map': torch.tensor(obs['occupancy_map'], dtype=torch.float32).unsqueeze(0).to(self.device),
                    }
                    tensor_observations.append(tensor_obs)
                
                # Forward pass through networks to get current policy and values
                curr_values = []
                action_probs = []
                constraint_losses = []
                entropies = []
                
                for i, obs in enumerate(tensor_observations):
                    # Encode state
                    z_t = self.encoder(obs)
                    
                    # Get state value
                    value = self.value_network(z_t).squeeze()
                    curr_values.append(value)
                    
                    # Create coarse mask
                    coarse_mask = create_coarse_mask(
                        obs['height_map'], obs['buffer'], self.bin_size, self.coarse_grid_size
                    )
                    
                    # Get coarse policy outputs
                    coarse_logits, coarse_probs = self.coarse_policy(z_t, coarse_mask)
                    
                    # Get action from batch
                    b, r, x, y = actions[i]
                    
                    # Find coarse grid cell for this action
                    W_c, L_c = self.coarse_grid_size
                    delta_x = self.W / W_c
                    delta_y = self.L / L_c
                    coarse_x = min(int(x // delta_x), W_c - 1)
                    coarse_y = min(int(y // delta_y), L_c - 1)
                    
                    # Get coarse action probability
                    coarse_prob = coarse_probs[0, b, r, coarse_x, coarse_y]
                    
                    # Get candidate region for fine policy
                    x_min = max(0, int((coarse_x * delta_x) - 1))
                    y_min = max(0, int((coarse_y * delta_y) - 1))
                    x_max = min(self.W, int(((coarse_x + 1) * delta_x) + 1))
                    y_max = min(self.L, int(((coarse_y + 1) * delta_y) + 1))
                    
                    # Get region data
                    region_data = {
                        'item_indices': torch.tensor([b], device=self.device),
                        'rotation': torch.tensor([r], device=self.device),
                        'x_min': torch.tensor([x_min], device=self.device),
                        'y_min': torch.tensor([y_min], device=self.device),
                        'x_max': torch.tensor([x_max], device=self.device),
                        'y_max': torch.tensor([y_max], device=self.device),
                        'items': obs['buffer'][0, b].view(1, 3)
                    }
                    
                    # Create fine mask
                    fine_mask = create_fine_mask(
                        obs['height_map'],
                        region_data['items'],
                        region_data['rotation'],
                        self.bin_size,
                        (region_data['x_min'], region_data['y_min'], 
                         region_data['x_max'], region_data['y_max'])
                    )
                    
                    # Get fine policy outputs
                    coarse_features = self.coarse_policy.get_feature_maps()
                    fine_logits, fine_probs = self.fine_policy(
                        z_t, region_data, coarse_features, None, fine_mask
                    )
                    
                    # Get heuristic policy outputs
                    heur_logits, heur_probs = heuristic_refinement(
                        obs['height_map'], fine_logits, region_data, fine_mask, self.bin_size,
                        method='greedy',
                        alpha_t=hyperparams['alpha'],
                        beta_t=hyperparams['beta'],
                        gamma=0.2
                    )
                    
                    # Ensemble results
                    final_logits, final_probs = self.meta_network(
                        fine_logits, heur_logits, fine_mask, z_t
                    )
                    
                    # Get fine action probability
                    fine_prob = final_probs[0, x, y]
                    
                    # Compute total probability
                    total_prob = coarse_prob * fine_prob
                    action_probs.append(total_prob)
                    
                    # Compute entropy
                    entropy = -torch.sum(final_probs * torch.log(final_probs + 1e-10))
                    entropies.append(entropy)
                    
                    # Compute constraint loss - penalize low support ratio
                    position = torch.tensor([[x, y]], device=self.device)
                    support_ratio = compute_support_ratio(
                        obs['height_map'],
                        region_data['items'],
                        region_data['rotation'],
                        position
                    )[0]
                    
                    # Penalize if support ratio is less than 0.8
                    constraint_loss = torch.max(
                        torch.tensor(0.0, device=self.device),
                        torch.tensor(0.8, device=self.device) - support_ratio
                    ) ** 2
                    constraint_losses.append(constraint_loss)
                
                # Stack values and probs
                curr_values = torch.stack(curr_values)
                action_probs = torch.stack(action_probs)
                constraint_losses = torch.stack(constraint_losses)
                entropies = torch.stack(entropies)
                
                # Compute advantages and returns
                advantages = self.advantage_estimator.compute_advantages(
                    rewards, values, torch.cat([values[1:], values[-1:].clone()]), dones
                )
                returns = self.advantage_estimator.compute_returns(advantages, values)
                
                # Normalize advantages
                advantages = normalize_advantages(advantages)
                
                # Compute probability ratio
                ratios = action_probs / (old_probs + 1e-10)
                
                # Compute surrogate objectives
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * advantages
                
                # Compute policy loss (negative because we want to maximize)
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                if self.value_clip:
                    values_clipped = values + torch.clamp(
                        curr_values - values, -self.value_clip, self.value_clip
                    )
                    v_loss1 = (curr_values - returns) ** 2
                    v_loss2 = (values_clipped - returns) ** 2
                    critic_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    critic_loss = 0.5 * ((curr_values - returns) ** 2).mean()
                
                # Compute entropy loss
                entropy_loss = entropies.mean()
                
                # Compute constraint loss
                constraint_loss = constraint_losses.mean()
                
                # Compute total loss
                total_loss = (
                    actor_loss 
                    + self.value_coef * critic_loss 
                    - self.entropy_coef * entropy_loss
                    + hyperparams['lambda'] * constraint_loss
                )
                
                # Compute KL divergence for early stopping
                kl_div = (old_probs * torch.log(old_probs / (action_probs + 1e-10) + 1e-10)).mean()
                
                # Backpropagate and update networks
                total_loss.backward()
                
                # Clip gradients
                all_params = []
                for model in [self.encoder, self.coarse_policy, self.fine_policy, 
                            self.meta_network, self.value_network]:
                    all_params.extend(list(model.parameters()))
                
                # Tính toán tổng gradient norm trước khi clip
                total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) 
                                                for p in all_params if p.grad is not None]), 2)
                epoch_metrics['grad_norm'] += total_norm.item()
                
                # Thực hiện gradient clipping với các parameter toàn cục
                torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                
                # Update networks
                self.encoder_optimizer.step()
                self.coarse_policy_optimizer.step()
                self.fine_policy_optimizer.step()
                self.meta_network_optimizer.step()
                self.value_network_optimizer.step()
                
                # Update epoch metrics
                epoch_metrics['actor_loss'] += actor_loss.item()
                epoch_metrics['critic_loss'] += critic_loss.item()
                epoch_metrics['entropy'] += entropy_loss.item()
                epoch_metrics['kl_div'] += kl_div.item()
                epoch_metrics['constraint_loss'] += constraint_loss.item()
                epoch_metrics['total_loss'] += total_loss.item()

            # Sau epoch, kiểm tra grad_norm để điều chỉnh learning rate tự động
            avg_grad_norm = epoch_metrics['grad_norm'] / len(batches)
            
            # Nếu gradient quá lớn, giảm learning rate
            if avg_grad_norm > self.max_grad_norm * 2:
                new_lr = current_lr * 0.8
                for optimizer in [self.encoder_optimizer, self.coarse_policy_optimizer,
                                self.fine_policy_optimizer, self.meta_network_optimizer,
                                self.value_network_optimizer]:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                if self.logger:
                    self.logger.log_scalar('lr', new_lr, self.step_counter)
            
            # Nếu gradient quá nhỏ, tăng learning rate
            elif avg_grad_norm < self.max_grad_norm * 0.1 and epoch > 0:
                new_lr = current_lr * 1.2
                for optimizer in [self.encoder_optimizer, self.coarse_policy_optimizer,
                                self.fine_policy_optimizer, self.meta_network_optimizer,
                                self.value_network_optimizer]:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                if self.logger:
                    self.logger.log_scalar('lr', new_lr, self.step_counter)
            
            # Average epoch metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= len(batches)
            
            # Update overall metrics
            for key in metrics:
                metrics[key] += epoch_metrics[key] / self.n_epochs
            
            # Điều chỉnh ngưỡng KL dựa trên số lượng bước huấn luyện
            adaptive_kl_target = self.target_kl
            if self.step_counter < 1000:
                # Cho phép KL cao hơn ở giai đoạn đầu
                adaptive_kl_target = self.target_kl * 3.0
                
            # Early stopping với ngưỡng KL thích ứng
            if epoch_metrics['kl_div'] > 1.5 * adaptive_kl_target:
                print(f"Early stopping at epoch {epoch+1}/{self.n_epochs} due to high KL divergence")
                break

        # Clear memory
        self.memory.clear()
        
        return metrics
    
    def save_models(self, path: str):
        """Save models to disk.
        
        Args:
            path: Path to save models
        """
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pt'))
        torch.save(self.coarse_policy.state_dict(), os.path.join(path, 'coarse_policy.pt'))
        torch.save(self.fine_policy.state_dict(), os.path.join(path, 'fine_policy.pt'))
        torch.save(self.meta_network.state_dict(), os.path.join(path, 'meta_network.pt'))
        torch.save(self.value_network.state_dict(), os.path.join(path, 'value_network.pt'))
        
        print(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load models from disk.
        
        Args:
            path: Path to load models from
        """
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pt')))
        self.coarse_policy.load_state_dict(torch.load(os.path.join(path, 'coarse_policy.pt')))
        self.fine_policy.load_state_dict(torch.load(os.path.join(path, 'fine_policy.pt')))
        self.meta_network.load_state_dict(torch.load(os.path.join(path, 'meta_network.pt')))
        self.value_network.load_state_dict(torch.load(os.path.join(path, 'value_network.pt')))
        
        print(f"Models loaded from {path}")
    
    def train(self, num_episodes: int, eval_interval: int = 10, save_path: Optional[str] = None):
        """Train the agent for a specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to train for
            eval_interval: Interval for evaluation and logging
            save_path: Path to save models (optional)
        """
        # Initialize metrics
        best_eval_reward = float('-inf')
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Run one episode
            episode_reward, steps, constraint_violations = self._run_episode(training=True)
            
            # Update metrics
            self.episode_rewards.append(episode_reward)
            self.constraint_violations.append(constraint_violations / max(1, steps))
            self.episode_counter += 1
            
            # Log episode metrics
            if self.logger:
                self.logger.log_scalar('episode_reward', episode_reward, self.episode_counter)
                self.logger.log_scalar('constraint_violations', constraint_violations / max(1, steps), self.episode_counter)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_violations = np.mean(self.constraint_violations[-10:])
                print(f"Episode {episode+1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.4f} | "
                      f"Avg Constraint Violations: {avg_violations:.4%} | "
                      f"Steps: {self.step_counter}")
            
            # Update dynamic hyperparameters
            if self.use_dynamic_hyperparams and episode > 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_violations = np.mean(self.constraint_violations[-10:])
                self.dynamic_hyperparams.update(avg_reward, avg_violations)
                
                # Log hyperparameters
                if self.logger:
                    hyperparams = self.dynamic_hyperparams.get_values()
                    for key, value in hyperparams.items():
                        self.logger.log_scalar(f'hyperparam/{key}', value, self.episode_counter)
            
            # Train networks if memory is full
            if self.memory.is_full():
                metrics = self.train_networks()
                
                # Log training metrics
                if self.logger:
                    for key, value in metrics.items():
                        self.logger.log_scalar(f'train/{key}', value, self.step_counter)
            
            # Evaluate and save models
            if (episode + 1) % eval_interval == 0:
                eval_reward = self._evaluate()
                
                if self.logger:
                    self.logger.log_scalar('eval/reward', eval_reward, self.episode_counter)
                
                print(f"Evaluation | Reward: {eval_reward:.4f}")
                
                # Save best model
                if save_path and eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.save_models(os.path.join(save_path, f'best_model'))
                
                # Save checkpoint
                if save_path:
                    self.save_models(os.path.join(save_path, f'checkpoint_{episode+1}'))
        
        # Training complete
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds.")
        print(f"Total steps: {self.step_counter}")
        
        # Save final model
        if save_path:
            self.save_models(os.path.join(save_path, 'final_model'))
    
    def _run_episode(self, training: bool = True) -> Tuple[float, int, int]:
        """Run a single episode.
        
        Args:
            training: Whether this is a training episode
            
        Returns:
            Tuple of (episode_reward, num_steps, constraint_violations)
        """
        # Reset environment
        observation, info = self.env.reset()
        
        # Track metrics
        episode_reward = 0
        steps = 0
        constraint_violations = 0
        done = False
        
        while not done:
            # Choose action
            action, action_prob, state_embed, debug_info = self.choose_action(observation)
            
            # Take action
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            
            # Update metrics
            episode_reward += reward
            steps += 1
            self.step_counter += 1
            
            # Track constraint violations
            if debug_info['constraint_violation'] > 0:
                constraint_violations += 1
            
            # Store transition in memory if training
            if training:
                self.memory.store(
                    observation=observation,
                    state=state_embed,
                    action=action,
                    prob=action_prob,
                    val=self.value_network(state_embed).item(),
                    reward=reward,
                    done=terminated or truncated
                )
            
            # Update observation
            observation = next_observation
            
            # Check if episode is done
            done = terminated or truncated
        
        return episode_reward, steps, constraint_violations
    
    def _evaluate(self, num_episodes: int = 5) -> float:
        """Evaluate the agent's performance.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Average reward across evaluation episodes
        """
        eval_rewards = []
        
        for _ in range(num_episodes):
            episode_reward, _, _ = self._run_episode(training=False)
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards)

def normalize_advantages(advantages):
    """Chuẩn hóa advantages một cách an toàn."""
    if advantages.numel() <= 1:
        return advantages  # Không cần chuẩn hóa nếu chỉ có 1 phần tử
    
    mean = advantages.mean()
    std = advantages.std()
    
    # Tránh chia cho 0 hoặc giá trị rất nhỏ
    if std.item() < 1e-8:
        return advantages - mean
    
    return (advantages - mean) / (std + 1e-8)