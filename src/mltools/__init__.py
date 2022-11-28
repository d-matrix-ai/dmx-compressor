import torch
from functools import partialmethod

torch.nn.Module.load_state_dict = partialmethod(
    torch.nn.Module.load_state_dict, strict=False
)
from . import (
    utils,
    numerical,
    sparse,
    approximate,
    functions,
    fx,
    corsair,
)
