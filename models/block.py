# transformer_block.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import logging

# Thiết lập logging để xem các dòng debug
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: int = 512, 
        dropout: float = 0.1
    ):
        """
        Khối Transformer thực hiện:
        - Self-Attention cho EMS và Item embeddings
        - Add & Norm
        - MLP cho EMS và Item embeddings
        - Add & Norm
        - Cross-Attention giữa EMS và Item embeddings
        - Add & Norm
        - MLP cuối cùng
        - Add & Norm
        """
        super(TransformerBlock, self).__init__()
        
        # Self-Attention cho EMS và Item embeddings
        self.self_attn_ems = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_item = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Layer Normalization sau Self-Attention
        self.norm1_ems = nn.LayerNorm(d_model)
        self.norm1_item = nn.LayerNorm(d_model)
        
        # MLP cho EMS và Item
        self.mlp_ems = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.mlp_item = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer Normalization sau MLP
        self.norm2_ems = nn.LayerNorm(d_model)
        self.norm2_item = nn.LayerNorm(d_model)
        
        # Cross-Attention: EMS attends to Items và ngược lại
        self.cross_attn_ems = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_item = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Layer Normalization sau Cross-Attention
        self.norm3_ems = nn.LayerNorm(d_model)
        self.norm3_item = nn.LayerNorm(d_model)
        
        # MLP cuối cùng cho EMS và Item
        self.mlp_final_ems = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.mlp_final_item = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer Normalization cuối cùng
        self.norm4_ems = nn.LayerNorm(d_model)
        self.norm4_item = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        ems: Tensor, 
        items: Tensor, 
        ems_mask: Tensor = None, 
        items_mask: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass cho khối Transformer.
        
        Args:
            ems (Tensor): EMS embeddings với kích thước [seq_len_ems, batch_size, d_model]
            items (Tensor): Item embeddings với kích thước [seq_len_item, batch_size, d_model]
            ems_mask (Tensor, optional): Mask cho EMS embeddings (True cho các phần thực)
            items_mask (Tensor, optional): Mask cho Item embeddings (True cho các phần thực)
        
        Returns:
            Tuple[Tensor, Tensor]: Cập nhật EMS và Item embeddings
        """
        logger.debug("Starting TransformerBlock forward pass")
        
        # --- Self-Attention cho EMS ---
        logger.debug("Self-Attention cho EMS")
        ems_self_attn_output, _ = self.self_attn_ems(ems, ems, ems, key_padding_mask=~ems_mask if ems_mask is not None else None)
        ems = ems + self.dropout(ems_self_attn_output)  # Skip connection
        ems = self.norm1_ems(ems)
        logger.debug("After Self-Attention và Add & Norm cho EMS")
        
        # --- Self-Attention cho Item ---
        logger.debug("Self-Attention cho Item")
        items_self_attn_output, _ = self.self_attn_item(items, items, items, key_padding_mask=~items_mask if items_mask is not None else None)
        items = items + self.dropout(items_self_attn_output)  # Skip connection
        items = self.norm1_item(items)
        logger.debug("After Self-Attention và Add & Norm cho Item")
        
        # --- MLP cho EMS ---
        logger.debug("MLP cho EMS")
        ems_mlp_output = self.mlp_ems(ems)
        ems = ems + self.dropout(ems_mlp_output)  # Skip connection
        ems = self.norm2_ems(ems)
        logger.debug("After MLP và Add & Norm cho EMS")
        
        # --- MLP cho Item ---
        logger.debug("MLP cho Item")
        items_mlp_output = self.mlp_item(items)
        items = items + self.dropout(items_mlp_output)  # Skip connection
        items = self.norm2_item(items)
        logger.debug("After MLP và Add & Norm cho Item")
        
        # --- Cross-Attention: EMS attends to Items ---
        logger.debug("Cross-Attention: EMS attends to Items")
        ems_cross_attn_output, _ = self.cross_attn_ems(
            ems,       # Query
            items,     # Key
            items,     # Value
            key_padding_mask=~items_mask if items_mask is not None else None
        )
        ems = ems + self.dropout(ems_cross_attn_output)  # Skip connection
        ems = self.norm3_ems(ems)
        logger.debug("After Cross-Attention và Add & Norm cho EMS")
        
        # --- Cross-Attention: Items attends to EMS ---
        logger.debug("Cross-Attention: Items attends to EMS")
        items_cross_attn_output, _ = self.cross_attn_item(
            items,      # Query
            ems,        # Key
            ems,        # Value
            key_padding_mask=~ems_mask if ems_mask is not None else None
        )
        items = items + self.dropout(items_cross_attn_output)  # Skip connection
        items = self.norm3_item(items)
        logger.debug("After Cross-Attention và Add & Norm cho Item")
        
        # --- MLP cuối cùng cho EMS ---
        logger.debug("MLP cuối cùng cho EMS")
        ems_final_mlp = self.mlp_final_ems(ems)
        ems = ems + self.dropout(ems_final_mlp)  # Skip connection
        ems = self.norm4_ems(ems)
        logger.debug("After MLP cuối cùng và Add & Norm cho EMS")
        
        # --- MLP cuối cùng cho Item ---
        logger.debug("MLP cuối cùng cho Item")
        items_final_mlp = self.mlp_final_item(items)
        items = items + self.dropout(items_final_mlp)  # Skip connection
        items = self.norm4_item(items)
        logger.debug("After MLP cuối cùng và Add & Norm cho Item")
        
        logger.debug("TransformerBlock forward pass completed")
        return ems, items
