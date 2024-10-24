# value_net.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class ValueNetwork(nn.Module):
    def __init__(
        self, 
        d_input: int = 128, 
        d_hidden: int = 256
    ):
        """
        Value Network for Bin Packing Problem.

        :param d_input: The size of the input embeddings.
        :param d_hidden: The size of the hidden layers.
        """
        super(ValueNetwork, self).__init__()
        
        # MLP for EMS list features
        self.ems_list_mlp = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        
        # MLP for buffer features
        self.buffer_mlp = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        
        # Final MLP for value prediction
        self.final_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
            nn.Tanh()
        )
        
    def forward(
        self, 
        ems_list_features: Tensor, 
        buffer_features: Tensor
    ) -> float:
        """
        Forward pass for the Value Network.

        :param ems_features: The embeddings of EMS list. # [batch_size, d_input]
        :param item_features: The embeddings of Item list. # [batch_size, d_input]
        """
        
        ems_list_out = self.ems_list_mlp(ems_list_features) # [batch_size, d_hidden]
        buffer_out = self.buffer_mlp(buffer_features) # [batch_size, d_hidden]
        
        combined = torch.cat((ems_list_out, buffer_out), dim=1) # [batch_size, 2*d_hidden]
    
        value = self.final_mlp(combined) # [batch_size, 1]

        return value
