# block.py

import torch.nn as nn
from torch import Tensor
from typing import Tuple

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: int = 256, 
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
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the Transformer Block.

        :param ems: EMS embeddings with shape [seq_len_ems, batch_size, d_model]
        :param items: Item embeddings with shape [seq_len_item, batch_size, d_model]
        :param ems_mask: Mask for EMS embeddings (True for real parts)

        :return: Tuple of updated EMS and Item embeddings
        """
        # --- Self-Attention cho EMS ---
        ems_self_attn_output, _ = self.self_attn_ems(
            ems, 
            ems, 
            ems,
            key_padding_mask=~ems_mask if ems_mask is not None else None
        )
        ems = ems + self.dropout(ems_self_attn_output)  # Skip connection
        ems = self.norm1_ems(ems)
        
        # --- Self-Attention cho Item ---
        items_self_attn_output, _ = self.self_attn_item(
            items, 
            items, 
            items, 
        )
        items = items + self.dropout(items_self_attn_output)  # Skip connection
        items = self.norm1_item(items)
        
        # --- MLP cho EMS ---
        ems_mlp_output = self.mlp_ems(ems)
        ems = ems + self.dropout(ems_mlp_output)  # Skip connection
        ems = self.norm2_ems(ems)
        
        # --- MLP cho Item ---
        items_mlp_output = self.mlp_item(items)
        items = items + self.dropout(items_mlp_output)  # Skip connection
        items = self.norm2_item(items)
        
        # --- Cross-Attention: EMS attends to Items ---
        ems_cross_attn_output, _ = self.cross_attn_ems(
            ems,       # Query
            items,     # Key
            items,     # Value
        )
        ems = ems + self.dropout(ems_cross_attn_output)  # Skip connection
        ems = self.norm3_ems(ems)
        
        # --- Cross-Attention: Items attends to EMS ---
        items_cross_attn_output, _ = self.cross_attn_item(
            items,      # Query
            ems,        # Key
            ems,        # Value
            key_padding_mask=~ems_mask if ems_mask is not None else None
        )
        items = items + self.dropout(items_cross_attn_output)  # Skip connection
        items = self.norm3_item(items)
        
        # --- MLP cuối cùng cho EMS ---
        ems_final_mlp = self.mlp_final_ems(ems)
        ems = ems + self.dropout(ems_final_mlp)  # Skip connection
        ems = self.norm4_ems(ems)
        
        # --- MLP cuối cùng cho Item ---
        items_final_mlp = self.mlp_final_item(items)
        items = items + self.dropout(items_final_mlp)  # Skip connection
        items = self.norm4_item(items)

        return ems, items
