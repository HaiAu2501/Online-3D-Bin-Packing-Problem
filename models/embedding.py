import torch.nn as nn
from torch import Tensor

class Embedding(nn.Module):
    def __init__(self, d_input: int, d_model: int):
        """
        Embedding for buffer or ems_list.
        
        :param d_input: input dimension
        :param d_model: model dimension

        - For buffer: d_input = 3
        - For ems_list: d_input = 6
        """
        super(Embedding, self).__init__()
        self.linear = nn.Linear(d_input, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the embedding layer.
        
        :param x: input tensor [batch_size, buffer_size, 3] or [batch_size, max_ems, 6]
        :return: embedded tensor [buffer_size, batch_size, d_model] or [max_ems, batch_size, d_model]
        """
        embedded = self.linear(x)  # [batch_size, buffer_size, d_model] or [batch_size, max_ems, d_model]
        embedded = embedded.transpose(0, 1)  # [buffer_size, batch_size, d_model] or [max_ems, batch_size, d_model]

        # NOTE: Transpose the batch_size to the second dimension to fit the Transformer architecture.
        return embedded