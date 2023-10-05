import torch
from functools import partialmethod

torch.nn.Module.load_state_dict = partialmethod(
    torch.nn.Module.load_state_dict, strict=False
)
from . import (
    utils,
    numerical,
    functional,
    sparse,
    fx,
    dmx,
    perf_proxy,
    layer_reconstruction,
)
