import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

__ALL__ = ["LeNet"]


ACT_FUNC_MAP = {
    "relu": nn.ReLU(inplace=True),
    "gelu": nn.GELU(),
}


class LeNet(nn.Module):
    def __init__(
        self, hidden_dims, input_dim=784, output_dim=10, act_func="relu"
    ) -> None:
        super(LeNet, self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.act_func = ACT_FUNC_MAP[act_func]
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.relu_(self.layers[0](x))
        for layer in self.layers[1:-1]:
            x = layer(x)
            x = self.act_func(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return F.log_softmax(x, dim=1)


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output
