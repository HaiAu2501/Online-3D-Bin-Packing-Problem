# networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class PolicyNetwork(nn.Module):
    """
    Mạng Policy sử dụng các lớp convolution.
    Đầu vào: tensor có 4 kênh (height_map và 3 kênh kích thước vật phẩm)
    Đầu ra: W x L x 6 (6 hướng xoay)
    """
    def __init__(self, input_channels: int, num_rotations: int = 6):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Pooling về kích thước 1x1
        self.fc = nn.Linear(128, num_rotations)  # Số lượng lớp đầu ra là số hướng xoay
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của mạng Policy.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x  # Output kích thước (batch_size, num_rotations)

class ValueNetwork(nn.Module):
    """
    Mạng Value nhận đầu vào là trạng thái và đầu ra là giá trị dự đoán.
    Sử dụng các lớp convolution tương tự PolicyNetwork.
    """
    def __init__(self, input_channels: int):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Pooling về kích thước 1x1
        self.fc = nn.Linear(128, 1)  # Đầu ra là một giá trị duy nhất
        self.activation = nn.Tanh()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của mạng Value.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.activation(x)
        return x  # Output kích thước (batch_size, 1)
