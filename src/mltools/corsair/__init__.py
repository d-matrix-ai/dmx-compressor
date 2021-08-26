import torch
from ..numerical import CastTo
from ..sparse import Sparsify
from ..approximate import Approximate
from .transform import Model
from .nn import *


# add new torch.nn modules for corsair
torch.nn.CastTo = CastTo
torch.nn.Sparsify = Sparsify
torch.nn.Approximate = Approximate

# overload existing torch.nn modules for corsair
torch.nn.Linear = Linear
torch.nn.Conv2d = Conv2d
torch.nn.AdaptiveAvgPool2d = AdaptiveAvgPool2d
torch.nn.MaxPool2d = MaxPool2d
torch.nn.BatchNorm2d = BatchNorm2d
torch.nn.LayerNorm = LayerNorm
torch.nn.Dropout = Dropout
torch.nn.Softmax = Softmax
torch.nn.ReLU = ReLU
torch.nn.ReLU6 = ReLU6
torch.nn.Tanh = Tanh
