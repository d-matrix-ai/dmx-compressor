import torch
import triton.language as tl
import sys

from .quantize import quantize_elemwise
from .mx import quantize_mx
from .bfp import quantize_bfp
import triton

'''
Code derived from https://github.com/microsoft/microxcaling/blob/b601f4e946fffd702ec8f2d41980b1788557d1b1/mx/cpp/funcs.cpp#L138
'''

def quantize_mx_func(A, scale_bits, ebits, mbits, emax, max_norm,
                          max_values, axis, block_size, flush_fp32_subnorms, rmode):
    assert axis < A.ndim, " quantize_mx_func axis exceeds input dimensions"
    assert A.device.type == "cuda", "A not on cuda device"
    assert max_values.device.type == "cuda", "max_values not on cuda device"

    BLOCK_SIZE = block_size

    return quantize_mx(
        A, scale_bits, ebits, mbits, emax, max_norm,
        max_values, axis, BLOCK_SIZE, flush_fp32_subnorms, rmode)

def quantize_bfp_func(A, scale_bits, ebits, mbits, max_norm,
                      max_values, symmetric, axis, block_size, rmode):
    assert axis < A.ndim, " quantize_mx_func axis exceeds input dimensions"
    assert A.device.type == "cuda", "A not on cuda device"
    assert max_values.device.type == "cuda", "max_values not on cuda device"

    BLOCK_SIZE = block_size

    return quantize_bfp(
        A, scale_bits, ebits, mbits, max_norm,
        max_values, symmetric, axis, BLOCK_SIZE, rmode)

'''
Code derived from https://github.com/microsoft/microxcaling/blob/b601f4e946fffd702ec8f2d41980b1788557d1b1/mx/cpp/funcs.cpp#L183
'''
@triton.jit
def quantize_elemwise_func(A_ptr, bits, exp_bits, max_norm,  n,
                           rmode, saturate_normals, allow_denorm, 
                           BLOCK_SIZE: tl.constexpr):
    assert A_ptr.device.type == "cuda", "A not on cuda device"
    assert bits <= 24, "quantize_elemwise with bits > 24 leads to negative shifts"

    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    return quantize_elemwise(tl.load(A_ptr + offsets, mask), bits, exp_bits, max_norm,
                                rmode, saturate_normals, allow_denorm)
