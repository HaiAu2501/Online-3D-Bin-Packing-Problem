import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

# Import các class từ các file .py tương ứng
from .transformer import BinPackingTransformer
from .value_net import ValueNetwork
from .policy_net import PolicyNetwork

class BinPackingModel(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_head: int = 8,
        n_layers: int = 3,
        d_feedforward: int = 512,
        d_action: int = 128, # Thay bằng kích thước không gian hành động
        d_hidden: int = 256,
    ):
        super(BinPackingModel, self).__init__()

        self.transformer = BinPackingTransformer(
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layers,
            d_feedforward=d_feedforward,
        )
        self.value_net = ValueNetwork(d_input=d_model, d_hidden=d_hidden)
        self.policy_net = PolicyNetwork(d_input=d_model, d_hidden=d_hidden, d_action=d_action)

    def forward(self, ems_list: Tensor, buffer: Tensor, action_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the entire model.

        :param ems_list: [batch_size, max_ems, 6]
        :param buffer: [batch_size, buffer_size, 3]
        :param action_mask: [batch_size, d_action]

        :return: Tuple of value and policy
        """
        ems_list_features, buffer_features = self.transformer(ems_list, buffer)
        value = self.value_net(ems_list_features, buffer_features)
        policy = self.policy_net(ems_list_features, buffer_features, action_mask)
        return value, policy

    def get_value(self, ems_list: Tensor, buffer: Tensor) -> Tensor:
        """
        Calculates the value of a state.

        :param ems_list: [batch_size, max_ems, 6]
        :param buffer: [batch_size, buffer_size, 3]

        :return: Value of the state [batch_size, 1]
        """
        ems_list_features, buffer_features = self.transformer(ems_list, buffer)
        value = self.value_net(ems_list_features, buffer_features)
        return value

    def get_policy(self, ems_list: Tensor, buffer: Tensor, action_mask: Tensor) -> Tensor:
        """
        Calculates the policy for a state.

        :param ems_list: [batch_size, max_ems, 6]
        :param buffer: [batch_size, buffer_size, 3]
        :param action_mask: [batch_size, d_action]

        :return: Policy for the state [batch_size, d_action]
        """
        ems_list_features, buffer_features = self.transformer(ems_list, buffer)
        policy = self.policy_net(ems_list_features, buffer_features, action_mask)
        return policy