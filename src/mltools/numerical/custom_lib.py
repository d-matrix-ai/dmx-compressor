#!/usr/bin/env python3
from typing import List, Tuple

from . import Quantize, DeQuantize

import torch

from torch.library import Library, impl




dmx = torch.library.Library("dmx", "DEF")

# Quantize
dmx.define("quantize(Tensor t) -> Tensor")

@impl(dmx, "quantize", "CompositeExplicitAutograd")
def quantize(t: torch.Tensor):
    return t

@impl(dmx, "quantize", "Meta")
def quantize_meta(t: torch.Tensor):
    return torch.empty_like(t)

# Dequantize
dmx.define("dequantize(Tensor t) -> Tensor")

@impl(dmx, "dequantize", "CompositeExplicitAutograd")
def dequantize(t: torch.Tensor):
    return t

@impl(dmx, "dequantize", "Meta")
def dequantize_meta(t: torch.Tensor):
    return torch.empty_like(t)

# Custom_Relu
dmx.define("custom_relu(Tensor t) -> Tensor")

@impl(dmx, "custom_relu", "CompositeExplicitAutograd")
def custom_relu(t: torch.Tensor):
    return t

@impl(dmx, "custom_relu", "Meta")
def custom_relu_meta(t: torch.Tensor):
    return torch.empty_like(t)

numerical_extra_lib = [
]

numerical_backend_legal_ops = []
