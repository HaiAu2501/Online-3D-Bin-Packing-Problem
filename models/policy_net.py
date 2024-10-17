# policy_net.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class PolicyNetwork(nn.Module):
    def __init__(
        self, 
        d_model: int = 128, 
        hidden_dim: int = 256, 
        **kwargs,
    ):
        """
        Policy Network for Bin Packing.
        
        Args:
            d_model (int): The size of the embeddings.
            hidden_dim (int): The size of the hidden layers in the MLP.
            action_dim (int): The size of the output action space (number of available actions).
        """
        super(PolicyNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        if 'W' not in kwargs or 'L' not in kwargs or 'num_rotations' not in kwargs or 'buffer_size' not in kwargs:
            raise ValueError("Please provide (W, L, num_rotations, buffer_size) or action_dim.")
        self.output_dim = kwargs.get('action_dim',  kwargs['W'] * kwargs['L'] * kwargs['num_rotations'] * kwargs['buffer_size'])

        self.ems_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.item_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Định nghĩa lớp Linear ở đây
        self.output_linear = nn.Linear(hidden_dim, self.output_dim)

        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, 
        ems_features: Tensor, 
        item_features: Tensor, 
        action_mask: Tensor
    ) -> Tensor:
        """
        Forward pass cho Policy Network.
        
        Args:
            ems_features (Tensor): [batch_size, d_model]
            item_features (Tensor): [batch_size, d_model]
            action_mask (Tensor): [batch_size, action_dim]
        
        Returns:
            Tensor: [batch_size, action_dim] - Phân phối xác suất cho các hành động
        """
        ems_out = self.ems_mlp(ems_features)  # [batch_size, hidden_dim]
        item_out = self.item_mlp(item_features)  # [batch_size, hidden_dim]
        combined = ems_out * item_out  # [batch_size, hidden_dim]
        combined = self.output_linear(combined)  # [batch_size, action_dim]
        probabilities = self.softmax(combined)  # [batch_size, action_dim]
        
        # Đảm bảo rằng action_mask cũng trên cùng thiết bị
        action_mask = action_mask.to(probabilities.device)
        
        masked_probabilities = probabilities * action_mask  # [batch_size, action_dim]
        
        # Tránh chia cho 0 bằng cách thêm một epsilon nhỏ
        masked_probabilities = masked_probabilities / (masked_probabilities.sum(dim=1, keepdim=True) + 1e-8)

        return masked_probabilities
