# transformer.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import logging

from block import TransformerBlock
from embedding import EMSEmbedding, BufferEmbedding


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BinPackingTransformer(nn.Module):
    def __init__(
        self, 
        d_model: int = 128, 
        nhead: int = 8, 
        num_layers: int = 3, 
        dim_feedforward: int = 512, 
        max_ems: int = 1000 # Should be W * L * H
    ):
        """
        Kiến trúc Transformer cho bài toán Bin Packing với hai đầu vào: EMS và Buffer Items.
        
        Args:
            d_model (int): Kích thước embedding.
            nhead (int): Số đầu attention.
            num_layers (int): Số lượng khối Transformer.
            dim_feedforward (int): Kích thước của MLP.
            max_ems (int): Độ dài tối đa cho danh sách EMS.
        """
        super(BinPackingTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_ems = max_ems
        
        # Embedding cho EMS và Buffer
        self.ems_embedding = EMSEmbedding(input_dim=6, d_model=d_model)
        self.buffer_embedding = BufferEmbedding(input_dim=3, d_model=d_model)
        
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
            ems_mask (Tensor, optional): Mask cho EMS embeddings (True cho các phần thực)
            buffer_mask (Tensor, optional): Mask cho Item embeddings (True cho các phần thực)
        
        Returns:
            Tuple[Tensor, Tensor]: EMS Features và Item Features
        """
        logger.debug("Starting forward pass of BinPackingTransformer")
        
        # Embedding
        ems_embedded = self.ems_embedding(ems_list)  # [num_ems, batch_size, d_model]
        items_embedded = self.buffer_embedding(buffer_list)  # [num_items, batch_size, d_model]
        
        logger.debug("After embedding (Positional Encoding đã bị loại bỏ)")
        
        # Padding EMS nếu cần
        batch_size, num_ems, _ = ems_list.size()
        if num_ems < self.max_ems:
            padding_size = self.max_ems - num_ems
            padding = torch.zeros(padding_size, batch_size, self.d_model).to(ems_embedded.device)
            ems_embedded = torch.cat((ems_embedded, padding), dim=0)  # [max_ems, batch_size, d_model]
            logger.debug(f"Padded EMS embeddings shape: {ems_embedded.shape}")
        
        # Tạo mask nếu chưa có
        if ems_mask is None:
            ems_mask = torch.zeros(batch_size, self.max_ems).bool().to(ems_embedded.device)
            ems_mask[:, :num_ems] = True  # True cho các phần thực, False cho padding
            logger.debug(f"Generated EMS mask shape: {ems_mask.shape}")
        
        # Transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            logger.debug(f"Processing TransformerBlock {idx+1}")
            ems_embedded, items_embedded = block(
                ems_embedded, 
                items_embedded, 
                ems_mask=~ems_mask,  # Đảo ngược mask cho PyTorch (True cho padding)
                items_mask=~buffer_mask if buffer_mask is not None else None
            )
            logger.debug(f"After TransformerBlock {idx+1}")
        
        # Aggregate thông tin từ EMS và Items
        logger.debug("Aggregating information from EMS and Items")
        # Lấy các EMS thực (không tính padding) và tính trung bình
        ems_real = ems_embedded[:num_ems, :, :]  # [num_real_ems, batch_size, d_model]
        ems_features = ems_real.mean(dim=0)  # [batch_size, d_model]
        items_features = items_embedded.mean(dim=0)  # [batch_size, d_model]
        logger.debug(f"EMS Features shape: {ems_features.shape}")
        logger.debug(f"Item Features shape: {items_features.shape}")
        
        logger.debug("Forward pass of BinPackingTransformer completed")
        return ems_features, items_features
