# value_net.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import logging

# Thiết lập logging
logger = logging.getLogger(__name__)

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
        
        logger.debug("ValueNetwork initialized")
        
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
        logger.debug("ValueNetwork forward pass started")
        
        # Đưa qua MLP riêng biệt
        ems_out = self.ems_mlp(ems_features)  # [batch_size, hidden_dim]
        item_out = self.item_mlp(item_features)  # [batch_size, hidden_dim]
        
        logger.debug("After MLPs - ems_out shape: {}, item_out shape: {}".format(ems_out.shape, item_out.shape))
        
        # Concatenation
        combined = torch.cat((ems_out, item_out), dim=1)  # [batch_size, hidden_dim * 2]
        logger.debug("After concatenation - combined shape: {}".format(combined.shape))
        
        # Đưa qua final MLP và áp dụng tanh
        value = self.final_mlp(combined)  # [batch_size, 1]
        logger.debug("After final MLP and Tanh - value shape: {}".format(value.shape))
        
        logger.debug("ValueNetwork forward pass completed")
        return value
