# transformer.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from .block import TransformerBlock
from .embedding import CombinedEmbedding
from .pooling import CombinedPooling


class BinPackingTransformer(nn.Module):
    def __init__(
        self, 
        d_model: int = 128, 
        n_head: int = 8, 
        n_layers: int = 3, 
        d_feedforward: int = 512, 
    ):
        """
        Transformer architecture for Bin Packing Problem with two inputs: EMS list and Item list in buffer.

        :param d_model: The size of the embeddings.
        :param n_head: The number of heads in the multiheadattention models (d_model % n_head == 0).
        :param n_layers: The number of sub-encoder-layers in the encoder.
        :param d_feedforward: The dimension of the feedforward network model.
        """
        
        super(BinPackingTransformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        
        # Embedding for EMS list and Item list
        self.ems_list_embedding = CombinedEmbedding(d_model=d_model, is_ems=True)
        self.buffer_embedding = CombinedEmbedding(d_model=d_model, is_ems=False)
        
        # Stack các TransformerBlock
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_feedforward) for _ in range(n_layers)
        ])
        self.combined_pooling = CombinedPooling(d_model)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(
        self, 
        ems_list: Tensor, 
        buffer: Tensor, 
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the entire Transformer architecture.

        :param ems_list: [batch_size, max_ems, 6]
        :param buffer_list: [batch_size, buffer_size, 3]

        :return: Tuple of EMS Features and Item Features
        """    
        # Embedding
        ems_list_embedded = self.ems_list_embedding(ems_list)  # [num_ems, batch_size, d_model]
        buffer_embedded = self.buffer_embedding(buffer)  # [buffer_size, batch_size, d_model]
        
        for block in self.transformer_blocks:
            ems_list_embedded, buffer_embedded = block(
                ems_list_embedded, 
                buffer_embedded,
            )

        ems_list_features = self.combined_pooling(ems_list_embedded)  # [batch_size, d_model]
        buffer_features = self.combined_pooling(buffer_embedded)  # [batch_size, d_model]

        return ems_list_features, buffer_features # [batch_size, d_model], [batch_size, d_model]

