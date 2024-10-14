# networks.py 

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class TransformerBlock(nn.Module):
    """
    Transformer block với cross-attention giữa height_map và item_sizes.
    """
    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, height_map_emb: torch.Tensor, item_sizes_emb: torch.Tensor):
        """
        Forward pass với cross-attention giữa height_map_emb và item_sizes_emb.

        :param height_map_emb: Tensor có shape (seq_len_height, batch_size, embed_dim)
        :param item_sizes_emb: Tensor có shape (seq_len_item, batch_size, embed_dim)
        :return: Tuple gồm height_map_emb và item_sizes_emb đã được cập nhật
        """
        # Cross-attention: item_sizes_emb chú ý đến height_map_emb
        attn_output, _ = self.cross_attn(item_sizes_emb, height_map_emb, height_map_emb)
        item_sizes_emb = self.layer_norm1(item_sizes_emb + attn_output)
        # Feed forward
        ff_output = self.feed_forward(item_sizes_emb)
        item_sizes_emb = self.layer_norm2(item_sizes_emb + ff_output)
        return height_map_emb, item_sizes_emb

class CombinedPolicyValueNetwork(nn.Module):
    """
    Mạng kết hợp sử dụng Transformer để xử lý height_map và item_sizes,
    sau đó đi vào PolicyNetwork và ValueNetwork.
    """
    def __init__(self, input_channels: int, embed_dim: int = 128, num_heads: int = 8, num_rotations: int = 6):
        super(CombinedPolicyValueNetwork, self).__init__()
        self.embed_dim = embed_dim
        self.num_rotations = num_rotations

        # Embedding layers
        self.height_map_conv = nn.Conv2d(input_channels, embed_dim, kernel_size=1)
        self.item_sizes_fc = nn.Linear(3, embed_dim)

        # Transformer Block
        self.transformer = TransformerBlock(embed_dim, num_heads)

        # PolicyNetwork
        self.policy_conv1 = nn.Conv2d(embed_dim, 32, kernel_size=3, padding=1)
        self.policy_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.policy_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.policy_adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Pooling về kích thước 1x1
        self.policy_fc = nn.Linear(128, num_rotations)  # Đầu ra: num_rotations

        # ValueNetwork
        self.value_conv1 = nn.Conv2d(embed_dim, 32, kernel_size=3, padding=1)
        self.value_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.value_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.value_adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Pooling về kích thước 1x1
        self.value_fc = nn.Linear(128, 1)  # Đầu ra: một giá trị duy nhất
        self.value_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass của CombinedPolicyValueNetwork.

        :param x: Tensor đầu vào có shape (batch_size, input_channels, W, L)
        :return: Tuple gồm policy_output (batch_size, W, L, num_rotations),
                 value_output (batch_size, 1)
        """
        batch_size, _, W, L = x.shape

        # Tách height_map và item_sizes từ input
        height_map = x[:, 0:1, :, :]  # (batch_size, 1, W, L)
        item_sizes = x[:, 1:4, 0, 0]  # (batch_size, 3)

        # Embed height_map
        height_map_emb = self.height_map_conv(height_map)  # (batch_size, embed_dim, W, L)
        height_map_emb = height_map_emb.view(batch_size, self.embed_dim, W * L)  # (batch_size, embed_dim, W*L)
        height_map_emb = height_map_emb.permute(2, 0, 1)  # (W*L, batch_size, embed_dim)

        # Embed item_sizes
        item_sizes_emb = self.item_sizes_fc(item_sizes)  # (batch_size, embed_dim)
        item_sizes_emb = item_sizes_emb.unsqueeze(0)  # (1, batch_size, embed_dim)

        # Transformer forward
        height_map_emb, item_sizes_emb = self.transformer(height_map_emb, item_sizes_emb)

        # Reshape height_map_emb trở lại (batch_size, embed_dim, W, L)
        height_map_emb = height_map_emb.permute(1, 2, 0)  # (batch_size, embed_dim, W*L)
        height_map_emb = height_map_emb.view(batch_size, self.embed_dim, W, L)  # (batch_size, embed_dim, W, L)

        # PolicyNetwork forward
        policy = F.relu(self.policy_conv1(height_map_emb))  # (batch_size, 32, W, L)
        policy = F.relu(self.policy_conv2(policy))  # (batch_size, 64, W, L)
        policy = F.relu(self.policy_conv3(policy))  # (batch_size, 128, W, L)
        policy = self.policy_adaptive_pool(policy)  # (batch_size, 128, 1, 1)
        policy = policy.view(batch_size, -1)  # (batch_size, 128)
        policy = self.policy_fc(policy)  # (batch_size, num_rotations)

        # Để tạo ra đầu ra W x L x 6, cần nhân bản các logits theo không gian
        policy = policy.unsqueeze(2).unsqueeze(3)  # (batch_size, num_rotations, 1, 1)
        policy = policy.expand(-1, -1, W, L)  # (batch_size, num_rotations, W, L)
        policy = policy.permute(0, 2, 3, 1)  # (batch_size, W, L, num_rotations)

        # ValueNetwork forward
        value = F.relu(self.value_conv1(height_map_emb))  # (batch_size, 32, W, L)
        value = F.relu(self.value_conv2(value))  # (batch_size, 64, W, L)
        value = F.relu(self.value_conv3(value))  # (batch_size, 128, W, L)
        value = self.value_adaptive_pool(value)  # (batch_size, 128, 1, 1)
        value = value.view(batch_size, -1)  # (batch_size, 128)
        value = self.value_fc(value)  # (batch_size, 1)
        value = self.value_activation(value)  # (batch_size, 1)

        return policy, value

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
