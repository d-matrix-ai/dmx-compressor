#!/usr/bin/env python3

import torch
from torch.fx import Graph


def qdq_attr(g: Graph, target: str, format: str):
    """Modifies a graph, adds get_attr node with given target,
     and wraps that access in a QdQ pair. It also assumes that
    corresponding scale and zero_point tensors exist.
    """
    attr = g.get_attr(target)
    attr_scale = g.get_attr(target + "_scale")
    attr_zero_point = g.get_attr(target + "_zero_point")
    attr_q = g.call_function(
        torch.ops.dmx.quantize,
        (attr, attr_scale, attr_zero_point, format),
    )
    attr_dq = g.call_function(
        torch.ops.dmx.dequantize, (attr_q, attr_scale, attr_zero_point)
    )
    return attr_dq
