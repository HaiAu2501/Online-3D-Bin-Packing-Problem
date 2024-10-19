# transformer.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from .block import TransformerBlock
from .embedding import EMSEmbedding, BufferEmbedding

class BinPackingTransformer(nn.Module):
    def __init__(
        self, 
        d_model: int = 128, 
        nhead: int = 8, 
        num_layers: int = 3, 
        dim_feedforward: int = 512, 
        max_ems: int = 100
    ):
        """
        Transformer architecture for Bin Packing Problem with two inputs: EMS list and Item list in buffer.

        :param d_model: The size of the embeddings.
        :param nhead: The number of heads in the multiheadattention models (d_model % nhead == 0).
        :param num_layers: The number of sub-encoder-layers in the encoder.
        :param dim_feedforward: The dimension of the feedforward network model.
        :param max_ems: The maximum number of EMS in the input list.
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(
        self, 
        ems_list: Tensor, 
        buffer_list: Tensor, 
        ems_mask: Tensor = None, 
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the entire Transformer architecture.

        :param ems_list: [batch_size, max_ems, 6]
        :param buffer_list: [batch_size, buffer_size, 3]
        :param ems_mask: Mask for EMS embeddings (True for real parts)

        :return: Tuple of EMS Features and Item Features
        """    
        # Embedding
        ems_embedded = self.ems_embedding(ems_list)  # [max_ems, batch_size, d_model]
        items_embedded = self.buffer_embedding(buffer_list)  # [num_items, batch_size, d_model]
        
        for block in self.transformer_blocks:
            ems_embedded, items_embedded = block(
                ems_embedded, 
                items_embedded, 
                ems_mask=~ems_mask,  # Đảo ngược mask cho PyTorch (True cho padding)
            )
        
        ems_features = ems_embedded.mean(dim=0)  # [batch_size, d_model]
        items_features = items_embedded.mean(dim=0)  # [batch_size, d_model]

        return ems_features, items_features

