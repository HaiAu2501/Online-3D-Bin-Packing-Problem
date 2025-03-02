import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Union


class ValueNetwork(nn.Module):
    """Value Network to estimate state values for PPO agent.
    
    This network takes encoded state representation and outputs a scalar value
    representing the expected cumulative reward starting from that state.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 256, 
        hidden_dims: List[int] = [256, 128, 64],
        activation: nn.Module = nn.ReLU()
    ):
        """Initialize the value network.
        
        Args:
            embedding_dim: Dimension of state embeddings from encoder
            hidden_dims: List of hidden layer dimensions
            activation: Activation function to use
        """
        super().__init__()
        
        # Create MLP layers
        layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim
        
        # Final layer outputs a scalar value
        layers.append(nn.Linear(prev_dim, 1))
        
        # Combine all layers
        self.value_network = nn.Sequential(*layers)
        
        # Initialize weights using orthogonal initialization (common for RL)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using orthogonal initialization."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute state values.
        
        Args:
            state_embedding: [batch_size, embedding_dim] tensor from encoder
            
        Returns:
            [batch_size, 1] tensor of state values
        """
        return self.value_network(state_embedding)
    
    def get_value(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """Get state value (with gradient detached for target computation).
        
        Args:
            state_embedding: [batch_size, embedding_dim] tensor from encoder
            
        Returns:
            [batch_size, 1] tensor of state values (detached)
        """
        with torch.no_grad():
            return self.forward(state_embedding).detach()


class AdvantageEstimator:
    """Utility class for computing advantages using GAE (Generalized Advantage Estimation)."""
    
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Initialize the advantage estimator.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE smoothing parameter
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute GAE advantages.
        
        Args:
            rewards: [batch_size] tensor of rewards
            values: [batch_size] tensor of state values
            next_values: [batch_size] tensor of next state values
            dones: [batch_size] tensor indicating episode termination
            
        Returns:
            [batch_size] tensor of advantages
        """
        # Compute TD errors
        deltas = rewards + self.gamma * next_values * (1.0 - dones) - values
        
        # Compute GAE
        advantages = torch.zeros_like(deltas)
        gae = 0.0
        
        # Traverse in reversed order for proper discounting
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
            
        return advantages
    
    def compute_returns(self, advantages: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute returns (advantage + value).
        
        Args:
            advantages: [batch_size] tensor of advantages
            values: [batch_size] tensor of state values
            
        Returns:
            [batch_size] tensor of returns
        """
        return advantages + values


import numpy as np  # Added import for initialization