import torch
import torch.nn as nn
from torch import Tensor

class CombinedEmbedding(nn.Module):
    def __init__(self, d_model: int, is_ems: bool = False):
        """
        Embedding for buffer or ems_list.

        :param d_model: model dimension
        :param is_ems: True if embedding for ems_list, False for buffer
        """
        super(CombinedEmbedding, self).__init__()
        self.is_ems = is_ems
        if self.is_ems:
            self.conv = nn.Conv1d(in_channels=6, out_channels=d_model, kernel_size=1)
        else:
            self.linear = nn.Linear(3, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the embedding layer.

        :param x: input tensor [batch_size, num_ems, 6] or [batch_size, buffer_size, 3]
        :return: embedded tensor [batch_size, d_model]
        """
        if self.is_ems:
            x = x.transpose(1, 2)  # [batch_size, 6, num_ems]
            conved = self.conv(x)  # [batch_size, d_model, num_ems]
            conved = conved.transpose(1, 2) # [batch_size, num_ems, d_model]
            conved = conved.transpose(0, 1)  # [num_ems, batch_size, d_model]
            return conved
        else:
            embedded = self.linear(x)  # [batch_size, buffer_size, d_model]
            embedded = embedded.transpose(0, 1)  # [buffer_size, batch_size, d_model]
            return embedded