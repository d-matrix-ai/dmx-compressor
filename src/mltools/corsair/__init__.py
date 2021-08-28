import torch
from ..numerical import CastTo
from ..sparse import Sparsify
from ..approximate import Approximate
from .transform import Model, aware
from . import nn

counterpart = {
    torch.nn.Linear: nn.Linear,
    torch.nn.Conv2d: nn.Conv2d,
    torch.nn.AdaptiveAvgPool2d: nn.AdaptiveAvgPool2d,
    torch.nn.MaxPool2d: nn.MaxPool2d,
    torch.nn.BatchNorm2d: nn.BatchNorm2d,
    torch.nn.LayerNorm: nn.LayerNorm,
    torch.nn.Dropout: nn.Dropout,
    torch.nn.Softmax: nn.Softmax,
    torch.nn.ReLU: nn.ReLU,
    torch.nn.ReLU6: nn.ReLU6,
    torch.nn.Tanh: nn.Tanh,
}