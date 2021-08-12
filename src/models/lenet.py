import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__ALL__ = ["LeNet"]


class LeNet(nn.Module):
    def __init__(self, hidden_dims, input_dim=784, output_dim=10) -> None:
        super(LeNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.intermediate_layers = nn.ModuleList(
            [
                nn.Linear(hidden_dims[i], hidden_dims[i + 1])
                for i in range(len(hidden_dims) - 1)
            ]
        )
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.act_func = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        x = self.act_func(x)
        for layer in self.intermediate_layers:
            x = layer(x)
            x = self.act_func(x)
        x = self.output_layer(x)
        return x 
