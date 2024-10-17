# value_net.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import logging

class ValueNetwork(nn.Module):
    def __init__(
        self, 
        d_model: int = 128, 
        hidden_dim: int = 256
    ):
        """
        Value Network cho bài toán Bin Packing.
        
        Args:
            d_model (int): Kích thước embedding.
            hidden_dim (int): Kích thước của các lớp ẩn trong MLP.
        """
        super(ValueNetwork, self).__init__()
        
        # MLP cho EMS Features
        self.ems_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # MLP cho Item Features
        self.item_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Final MLP để tạo ra giá trị
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
    def forward(
        self, 
        ems_features: Tensor, 
        item_features: Tensor
    ) -> Tensor:
        """
        Forward pass cho Value Network.
        
        Args:
            ems_features (Tensor): [batch_size, d_model]
            item_features (Tensor): [batch_size, d_model]
        
        Returns:
            Tensor: [batch_size, 1] - Giá trị dự đoán cho mỗi sample
        """
        
        # Đưa qua MLP riêng biệt
        ems_out = self.ems_mlp(ems_features)  # [batch_size, hidden_dim]
        item_out = self.item_mlp(item_features)  # [batch_size, hidden_dim]
        
        # Concatenation
        combined = torch.cat((ems_out, item_out), dim=1)  # [batch_size, hidden_dim * 2]
        
        # Đưa qua final MLP và áp dụng tanh
        value = self.final_mlp(combined)  # [batch_size, 1]

        return value
