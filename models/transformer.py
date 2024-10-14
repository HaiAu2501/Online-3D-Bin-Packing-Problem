# transformer.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import logging
from block import TransformerBlock

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class EMSEmbedding(nn.Module):
    def __init__(self, input_dim: int = 6, d_model: int = 128):
        """
        Embedding cho danh sách EMS.
        """
        super(EMSEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, d_model)
        logger.debug("EMSEmbedding initialized")
    
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
        logger.debug(f"EMS embedded shape: {embedded.shape}")
        return embedded

class BufferEmbedding(nn.Module):
    def __init__(self, input_dim: int = 3, d_model: int = 128):
        """
        Embedding cho danh sách Item trong buffer.
        """
        super(BufferEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, d_model)
        logger.debug("BufferEmbedding initialized")
    
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
        logger.debug(f"Buffer embedded shape: {embedded.shape}")
        return embedded

class BinPackingTransformer(nn.Module):
    def __init__(
        self, 
        d_model: int = 128, 
        nhead: int = 8, 
        num_layers: int = 3, 
        dim_feedforward: int = 512, 
        max_len: int = 5000  # Bạn có thể loại bỏ tham số này nếu không còn sử dụng
    ):
        """
        Kiến trúc Transformer cho bài toán Bin Packing với hai đầu vào: EMS và Buffer Items.
        
        Args:
            d_model (int): Kích thước embedding.
            nhead (int): Số đầu attention.
            num_layers (int): Số lượng khối Transformer.
            dim_feedforward (int): Kích thước của MLP.
            max_len (int): Độ dài tối đa cho Positional Encoding (có thể loại bỏ nếu không sử dụng).
        """
        super(BinPackingTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Embedding cho EMS và Buffer
        self.ems_embedding = EMSEmbedding(input_dim=6, d_model=d_model)
        self.buffer_embedding = BufferEmbedding(input_dim=3, d_model=d_model)
        
        # Positional Encoding đã bị loại bỏ
        # self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Stack các TransformerBlock
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward) for _ in range(num_layers)
        ])
        logger.debug(f"{num_layers} TransformerBlocks initialized")
        
    def forward(
        self, 
        ems_list: Tensor, 
        buffer_list: Tensor, 
        ems_mask: Tensor = None, 
        buffer_mask: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass cho toàn bộ kiến trúc Transformer.
        
        Args:
            ems_list (Tensor): [batch_size, num_ems, 6]
            buffer_list (Tensor): [batch_size, num_items, 3]
            ems_mask (Tensor, optional): Mask cho EMS embeddings
            buffer_mask (Tensor, optional): Mask cho Item embeddings
        
        Returns:
            Tuple[Tensor, Tensor]: EMS Features và Item Features
        """
        logger.debug("Starting forward pass of BinPackingTransformer")
        
        # Embedding
        ems_embedded = self.ems_embedding(ems_list)  # [num_ems, batch_size, d_model]
        items_embedded = self.buffer_embedding(buffer_list)  # [num_items, batch_size, d_model]
        
        logger.debug("After embedding (Positional Encoding đã bị loại bỏ)")
        
        # Stack các TransformerBlock
        for idx, block in enumerate(self.transformer_blocks):
            logger.debug(f"Processing TransformerBlock {idx+1}")
            ems_embedded, items_embedded = block(
                ems_embedded, 
                items_embedded, 
                ems_mask=ems_mask, 
                items_mask=buffer_mask
            )
            logger.debug(f"After TransformerBlock {idx+1}")
        
        # Aggregate thông tin từ EMS và Items nếu cần (ví dụ: lấy mean)
        # Nếu bạn không cần aggregation, có thể bỏ qua bước này và trả về các embeddings đã qua các khối Transformer
        logger.debug("Aggregating information from EMS and Items")
        ems_features = ems_embedded.mean(dim=0)  # [batch_size, d_model]
        items_features = items_embedded.mean(dim=0)  # [batch_size, d_model]
        logger.debug(f"EMS Features shape: {ems_features.shape}")
        logger.debug(f"Item Features shape: {items_features.shape}")
        
        logger.debug("Forward pass of BinPackingTransformer completed")
        return ems_features, items_features

# Example usage (có thể được đặt trong một file khác hoặc trong phần testing)
if __name__ == "__main__":
    # Giả sử batch_size = 2, num_ems = 10, num_items = 15
    batch_size = 2
    num_ems = 10
    num_items = 15
    d_model = 128
    max_len = 5000
    
    # Tạo dummy inputs
    ems_input = torch.randint(0, 100, (batch_size, num_ems, 6)).float()
    buffer_input = torch.randint(0, 50, (batch_size, num_items, 3)).float()
    
    # Tạo mask (optional)
    ems_mask = None  # Nếu cần, có thể tạo mask ở đây
    buffer_mask = None  # Nếu cần, có thể tạo mask ở đây
    
    # Khởi tạo mô hình
    model = BinPackingTransformer(d_model=d_model, nhead=8, num_layers=3, dim_feedforward=512, max_len=max_len)
    
    # Forward pass
    ems_features, item_features = model(ems_input, buffer_input, ems_mask, buffer_mask)
    
    logger.debug(f"EMS Features shape: {ems_features.shape}")  # [batch_size, d_model]
    logger.debug(f"Item Features shape: {item_features.shape}")  # [batch_size, d_model]
