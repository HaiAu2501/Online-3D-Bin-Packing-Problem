# policy_net.py

import torch.nn as nn
from torch import Tensor

class PolicyNetwork(nn.Module):
    def __init__(
        self,
        d_action: int = None,
        d_input: int = 128, 
        d_hidden: int = 256, 
    ):
        """
        Policy Network for Bin Packing.
        
        d_input: The size of the input embeddings.
        d_hidden: The size of the hidden layers.
        d_action: The size of the action embeddings.
        """
        super(PolicyNetwork, self).__init__()

        self.ems_list_mlp = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )

        self.buffer_mlp = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        self.output_linear = nn.Linear(d_hidden, d_action)

        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, 
        ems_list_features: Tensor,
        buffer_features: Tensor,
        action_mask: Tensor
    ) -> Tensor:
        """
        Forward pass for the Policy Network.

        :param ems_list_features: The embeddings of EMS list. # [batch_size, d_input]
        :param buffer_features: The embeddings of Item list. # [batch_size, d_input]
        :param action_mask: The mask for valid actions. # [batch_size, d_action]

        :return: The probabilities of each action. # [batch_size, d_action]
        """

        ems_list_out = self.ems_list_mlp(ems_list_features) # [batch_size, d_hidden]
        buffer_out = self.buffer_mlp(buffer_features) # [batch_size, d_hidden]

        combined = ems_list_out * buffer_out # [batch_size, d_hidden] (*)
        combined = self.output_linear(combined) # [batch_size, d_action]
        probabilities = self.softmax(combined) # [batch_size, d_action] 
        masked_probabilities = probabilities * action_mask  # [batch_size, d_action]

        return masked_probabilities

        
