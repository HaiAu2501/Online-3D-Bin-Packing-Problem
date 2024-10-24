# block.py

import torch.nn as nn
from torch import Tensor
from typing import Tuple

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_head: int, 
        d_feedforward: int = 256, 
        dropout: float = 0.1
    ):
        """
        Architecture of a single Transformer Block in the Bin Packing Transformer.

        :param d_model: Dimension of the input embeddings
        :param n_head: Number of attention heads
        :param d_feedforward: Dimension of the feedforward network
        :param dropout: Dropout rate

        Layers:
        1: Self-Attention for 'ems_list' and 'buffer' embeddings
        2: Add & Norm
        3: MLP for 'ems_list' and 'buffer' embeddings
        4: Add & Norm
        5: Cross-Attention between 'ems_list' and 'buffer' embeddings
        6: Add & Norm
        7: Final MLP
        8: Add & Norm
        """
        super(TransformerBlock, self).__init__()
        
        # Self-Attention for 'ems_list' and 'buffer' embeddings
        self.self_attn_ems_list = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.self_attn_buffer = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        
        # Layer Normalization
        self.norm1_ems_list = nn.LayerNorm(d_model)
        self.norm1_buffer = nn.LayerNorm(d_model)
        
        # MLP for 'ems_list' and 'buffer' embeddings
        self.mlp_ems_list = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Linear(d_feedforward, d_model)
        )
        self.mlp_buffer = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Linear(d_feedforward, d_model)
        )
        
        # Layer Normalization
        self.norm2_ems_list = nn.LayerNorm(d_model)
        self.norm2_buffer = nn.LayerNorm(d_model)
        
        # Cross-Attention between 'ems_list' and 'buffer' embeddings
        self.cross_attn_ems_list = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.cross_attn_buffer = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        
        # Layer Normalization
        self.norm3_ems_list = nn.LayerNorm(d_model)
        self.norm3_buffer = nn.LayerNorm(d_model)
        
        # Final MLP
        self.mlp_final_ems_list = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Linear(d_feedforward, d_model)
        )
        self.mlp_final_buffer = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Linear(d_feedforward, d_model)
        )
        
        # Layer Normalization
        self.norm4_ems_list = nn.LayerNorm(d_model)
        self.norm4_buffer = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        ems_list: Tensor, 
        buffer: Tensor, 
        ems_mask: Tensor = None, 
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the Transformer Block.

        :param ems_list: Tensor of EMS embeddings [max_ems, batch_size, d_model]
        :param buffer: Tensor of buffer embeddings [buffer_size, batch_size, d_model]
        :param ems_mask: Mask for padding in EMS embeddings

        :return: Tuple of updated EMS and Item embeddings
        """
        # --- Self-Attention cho EMS ---
        ems_attn_output, _ = self.self_attn_ems_list(ems_list, ems_list, ems_list, key_padding_mask=ems_mask)
        ems_list = ems_list + self.dropout(ems_attn_output)  # Residual connection
        ems_list = self.norm1_ems_list(ems_list)  # Layer normalization

        ems_mlp_output = self.mlp_ems_list(ems_list)
        ems_list = ems_list + self.dropout(ems_mlp_output)  # Residual connection
        ems_list = self.norm2_ems_list(ems_list)  # Layer normalization

        # --- Self-Attention cho Buffer ---
        buffer_attn_output, _ = self.self_attn_buffer(buffer, buffer, buffer)
        buffer = buffer + self.dropout(buffer_attn_output)  # Residual connection
        buffer = self.norm1_buffer(buffer)  # Layer normalization

        buffer_mlp_output = self.mlp_buffer(buffer)
        buffer = buffer + self.dropout(buffer_mlp_output)  # Residual connection
        buffer = self.norm2_buffer(buffer)  # Layer normalization

        # --- Cross-Attention giữa EMS và Buffer ---
        cross_attn_ems_output, _ = self.cross_attn_ems_list(ems_list, buffer, buffer)
        ems_list = ems_list + self.dropout(cross_attn_ems_output)  # Residual connection
        ems_list = self.norm3_ems_list(ems_list)  # Layer normalization

        cross_attn_buffer_output, _ = self.cross_attn_buffer(buffer, ems_list, ems_list)
        buffer = buffer + self.dropout(cross_attn_buffer_output)  # Residual connection
        buffer = self.norm3_buffer(buffer)  # Layer normalization

        # --- Final MLP ---
        ems_final_mlp = self.mlp_final_ems_list(ems_list)
        ems_list = ems_list + self.dropout(ems_final_mlp)  # Residual connection
        ems_list = self.norm4_ems_list(ems_list)  # Layer normalization

        buffer_final_mlp = self.mlp_final_buffer(buffer)
        buffer = buffer + self.dropout(buffer_final_mlp)  # Residual connection
        buffer = self.norm4_buffer(buffer)  # Layer normalization

        return ems_list, buffer