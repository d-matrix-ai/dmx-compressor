import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.cpp_extension import load
import os

current_path = os.path.dirname(os.path.realpath(__file__))
quant_cpu = load(
    name="quant_cpu",
    sources=[
        os.path.join(current_path, "quant_cpu/quant_cpu.cpp"),
        os.path.join(current_path, "quant_cpu/bit_helper.cpp"),
        os.path.join(current_path, "quant_cpu/sim_helper.cpp"),
    ],
)

if torch.cuda.is_available():
    quant_cuda = load(
        name="quant_cuda",
        sources=[
            os.path.join(current_path, "quant_cuda/quant_cuda.cpp"),
            os.path.join(current_path, "quant_cuda/bit_helper.cu"),
            os.path.join(current_path, "quant_cuda/sim_helper.cu"),
            os.path.join(current_path, "quant_cuda/block_kernel.cu"),
            os.path.join(current_path, "quant_cuda/float_kernel.cu"),
            os.path.join(current_path, "quant_cuda/fixed_point_kernel.cu"),
            os.path.join(current_path, "quant_cuda/quant.cu"),
        ],
    )
else:
    quant_cuda = quant_cpu

__all__ = ["fixed_point_quantize", "block_quantize", "float_quantize"]


def assert_wl_fl(wl, fl, stage=""):
    if wl == -1 and fl != -1:
        raise ValueError("fixed point {} wl {}, fl {}".format(stage, wl, fl))


def get_module(x):
    if x.is_cuda:
        quant_module = quant_cuda
    else:
        quant_module = quant_cpu
    return quant_module


def fixed_point_quantize(x, wl, fl, clamp=True, symmetric=False, rounding="stochastic"):
    """
    Quantize a single precision Floating Point into low-precision Fixed Point

    Args:
        - :param: `x` (torch.Tensor) :  the single precision number to be quantized
        - :param: `wl` (int) : word length of the fixed point number being simulated
        - :param: `fl` (int) : fractional length of the fixed point number being simulated
        - :param: `clamp` (bool, optional) : clamp input numbers into representable range. if false,
                  the quantization will only simulate the effect on precision
        - :param: `symmetric` (bool, optional) : discard the minimum representable number to make the representable
                  range symmetric
        - :param: `rounding` (string) : rounding mode, \"stochastic\" or \"nearest\" (default: \"stochastic\")

    Returns:
        - a quantized low-precision block floating point number (torch.Tensor)
    """
    assert isinstance(x, torch.Tensor)
    assert rounding in ["stochastic", "nearest"]
    assert_wl_fl(wl, fl)
    quant_module = get_module(x)
    if rounding == "nearest":
        out = quant_module.fixed_point_quantize_nearest(
            x.contiguous(), wl, fl, clamp, symmetric
        )
    elif rounding == "stochastic":
        out = quant_module.fixed_point_quantize_stochastic(
            x.contiguous(), wl, fl, clamp, symmetric
        )
    return out


def block_quantize(x, wl, dim=-1, rounding="stochastic"):
    """
    Quantize a single precision Floating Point into low-precision Block Floating Point

    Args:
        - :param: `x` (torch.Tensor) :  the single precision number to be quantized
        - :param: `wl` (int) : word length of the block floating point number being simulated
        - :param: `rounding` (string) : rounding mode, \"stochastic\" or \"nearest\"

    Returns:
        - a quantized low-precision block floating point number (torch.Tensor)
    """
    assert isinstance(
        x, torch.Tensor
    ), "x is not a single precision Floating Point Tensor"
    assert rounding in [
        "stochastic",
        "nearest",
        "down",
        "up",
    ], "invalid rounding mode, {}".format(rounding)
    quant_module = get_module(x)
    if rounding == "nearest":
        out = quant_module.block_quantize_nearest(x.contiguous(), wl, dim)
    elif rounding == "stochastic":
        out = quant_module.block_quantize_stochastic(x.contiguous(), wl, dim)
    elif rounding == "down":
        out = quant_module.block_quantize_down(x.contiguous(), wl, dim)
    elif rounding == "up":
        out = quant_module.block_quantize_up(x.contiguous(), wl, dim)
    return out


def float_quantize(x, exp, man, bias=None, rounding="stochastic"):
    """
    Quantize a single precision Floating Point into low-precision Floating Point

    Args:
        - :attr: `x` (torch.Tensor) : the single precision number(torch.Tensor) to be quantized
        - :attr: `exp` (int) : number of bits allocated for exponent
        - :attr: `man` (int) : number of bits allocated for mantissa, not counting the virtual bit
        - :attr: `bias` (Optional[int]) : exponent bias
        - :attr: `rounding` (string) : rounding mode, \"stochastic\" or \"nearest\"

    Returns:
        - a quantized low-precision floating point number (torch.Tensor)
    """
    assert isinstance(
        x, torch.Tensor
    ), "x is not a single precision Floating Point Tensor"
    assert rounding in ["stochastic", "nearest"], "invalid rounding mode, {}".format(
        rounding
    )
    quant_module = get_module(x)
    if bias is None:
        bias = 2 ** (exp - 1) - 1
    if rounding == "nearest":
        out = quant_module.float_quantize_nearest(x.contiguous(), man, exp, bias)
    elif rounding == "stochastic":
        out = quant_module.float_quantize_stochastic(x.contiguous(), man, exp, bias)
    return out
