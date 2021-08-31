import torch
from . import (
    utils,
    numerical,
    sparse,
    approximate,
    functions,
    corsair,
    data,
    models,
    dmir,
)
torch.nn.Module = corsair.nn.Module