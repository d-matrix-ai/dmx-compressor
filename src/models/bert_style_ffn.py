import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional


__ALL__ = ["BERTStyleFFN"]


class Block(nn.Module):
    def __init__(self, d=1024, d_int=4096, act_func="gelu") -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d, d_int),
            nn.ReLU(inplace=True) if act_func=='relu' else nn.GELU(),
            nn.Linear(d_int, d),
            nn.Dropout(0.1),
        )
        self.ln = nn.LayerNorm(d, eps=1e-12)

    def forward(self, x):
        return self.ln(x + self.ffn(x))


class BERTStyleFFN(nn.Module):
    def __init__(self, depth=1, width=1024, act_func="gelu"):
        super().__init__()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, width),
            nn.Dropout(0.1),
            nn.LayerNorm(width, eps=1e-12),
        )
        self.body = nn.ModuleList(
            [Block(d=width, d_int=width*4, act_func=act_func) for _ in range(depth)]
        )
        self.tail = nn.Sequential(
            nn.Linear(width, 10),
        )

    def forward(self, x):
        x = self.head(x)
        for layer in self.body:
            x = layer(x)
        x = self.tail(x)
        return F.log_softmax(x, dim=1)