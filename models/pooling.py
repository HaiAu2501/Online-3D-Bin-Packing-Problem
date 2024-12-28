import torch
import torch.nn as nn
from torch import Tensor

class CombinedPooling(nn.Module):
    def __init__(self, d_model):
        super(CombinedPooling, self).__init__()
        self.attn = nn.Linear(d_model, 1)
        self.pooling_linear = nn.Linear(3*d_model, d_model)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: [batch_size, d_model]
        :return: [batch_size, d_model]
        """
        mean_pool = x.mean(dim=0)  # [d_model]
        max_pool, _ = x.max(dim=0)  # [d_model]
        attn_weights = torch.softmax(self.attn(x).squeeze(-1), dim=0)  # [sequence_length, batch_size]
        attn_pool = torch.sum(x * attn_weights.unsqueeze(-1), dim=0)  # [batch_size, d_model]
        combined = torch.cat([mean_pool, max_pool, attn_pool], dim=1)  # [batch_size, 3*d_model]
        combined = self.pooling_linear(combined) # [batch_size]
        return combined