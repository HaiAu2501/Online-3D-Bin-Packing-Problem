import torch.nn as nn
from torch import Tensor

class EMSEmbedding(nn.Module):
    def __init__(self, input_dim: int = 6, d_model: int = 128):
        """
        Embedding cho danh sách EMS.
        """
        super(EMSEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, d_model)
    
    def forward(self, ems_list: Tensor) -> Tensor:
        """
        Chuyển đổi EMS list thành embeddings.
        
        Args:
            ems_list (Tensor): [batch_size, num_ems, 6]
        
        Returns:
            Tensor: [num_ems, batch_size, d_model]
        """
        embedded = self.linear(ems_list)  # [batch_size, num_ems, d_model]
        embedded = embedded.transpose(0, 1)  # [num_ems, batch_size, d_model]
        return embedded

class BufferEmbedding(nn.Module):
    def __init__(self, input_dim: int = 3, d_model: int = 128):
        """
        Embedding cho danh sách Item trong buffer.
        """
        super(BufferEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, d_model)
    
    def forward(self, buffer_list: Tensor) -> Tensor:
        """
        Chuyển đổi buffer list thành embeddings.
        
        Args:
            buffer_list (Tensor): [batch_size, num_items, 3]
        
        Returns:
            Tensor: [num_items, batch_size, d_model]
        """
        embedded = self.linear(buffer_list)  # [batch_size, num_items, d_model]
        embedded = embedded.transpose(0, 1)  # [num_items, batch_size, d_model]
        return embedded