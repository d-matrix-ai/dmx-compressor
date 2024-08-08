'''
Triton implementation for S, N, D, U rounding of BFP type
'''

import torch
import triton.language as tl

from dmx.compressor.numerical.format import Format

default_format = Format.from_shorthand("BFP[8|8]{64}(SN)")

import triton
from .common import (get_grid,
                    round_bitwise_stochastic, clip_max_exponent,
                    round_bitwise_nearest, round_bitwise_down,
                    round_bitwise_up,
                    ROUNDING_MODE_INT)

def get_cuda_autotune_config():
    return [
        triton.Config(kwargs={'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config(kwargs={'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config(kwargs={'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config(kwargs={'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=2),
        triton.Config(kwargs={'BLOCK_SIZE': 2048}, num_warps=2),
        triton.Config(kwargs={'BLOCK_SIZE': 32}, num_warps=4),
        triton.Config(kwargs={'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config(kwargs={'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config(kwargs={'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config(kwargs={'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config(kwargs={'BLOCK_SIZE': 32}, num_warps=8),
        triton.Config(kwargs={'BLOCK_SIZE': 64}, num_warps=8),
        triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=8),
        triton.Config(kwargs={'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config(kwargs={'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config(kwargs={'BLOCK_SIZE': 2048}, num_warps=8),
    ]


@triton.autotune(configs=get_cuda_autotune_config(),
  key=['M', 'N'] # the two above configs will be evaluated anytime
                 # the value of x_size changes
)
@triton.jit
def quantize_bfp_kernel_stochastic (input_ptr, max_ptr, n,
                        output_ptr,
                        M, N,
                        symmetric, elem_mbits,
                        BLOCK_SIZE: tl.constexpr,
                        # **META
                        ):
    # BLOCK_SIZE = META['BLOCK_SIZE']
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    rand_ints = tl.randint(0, offsets)
    max_entry = tl.load(max_ptr + offsets, mask)
    max_entry_bits = max_entry.to(tl.int32, bitcast=True)

    a = tl.load(input_ptr + offsets, mask)
    conditon_tensor = ((a == -max_entry) & ((max_entry_bits >> 16 << 25) == 0xFE000000)) & (~symmetric)
    max_entry_bits = tl.where(conditon_tensor, (max_entry_bits >> 23) + 1 << 23, max_entry_bits)
    
    max_exp = max_entry_bits << 1 >> 24 << 23
    base_float = 6 * max_exp.to(tl.float32, bitcast=True)

    target_rebase = a + base_float
    target_bits = target_rebase.to(tl.int32, bitcast=True)
    quantized = round_bitwise_stochastic(target_bits, rand_ints, elem_mbits)
    quantize_float = quantized.to(tl.float32, bitcast=True) - base_float

    quantize_bits = quantize_float.to(tl.int32, bitcast=True)
    clip_quantize = clip_max_exponent(quantize_bits, max_exp, elem_mbits-2)
    quantize_float = clip_quantize.to(tl.float32, bitcast=True)
    tl.store(output_ptr + offsets, quantize_float, mask)

@triton.autotune(configs=get_cuda_autotune_config(),
  key=['M', 'N'] # the two above configs will be evaluated anytime
                 # the value of x_size changes
)
@triton.jit
def quantize_bfp_kernel_nearest (input_ptr, max_ptr, n,
                        output_ptr,
                        M, N,
                        symmetric, elem_mbits,
                        BLOCK_SIZE: tl.constexpr,
                        # **META
                        ):
    # BLOCK_SIZE = META['BLOCK_SIZE']
    
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    max_entry = tl.load(max_ptr + offsets, mask)
    max_entry_bits = max_entry.to(tl.int32, bitcast=True)
    
    a = tl.load(input_ptr + offsets, mask)
    conditon_tensor = ((a == -max_entry) & ((max_entry_bits >> 16 << 25) == 0xFE000000)) & (~symmetric)
    max_entry_bits = tl.where(conditon_tensor, (max_entry_bits >> 23) + 1 << 23, max_entry_bits)
    
    max_exp = max_entry_bits << 1 >> 24 << 23
    base_float = 6 * max_exp.to(tl.float32, bitcast=True)

    target_rebase = a + base_float
    target_bits = target_rebase.to(tl.int32, bitcast=True)
    quantized = round_bitwise_nearest(target_bits, elem_mbits)
    quantize_float = quantized.to(tl.float32, bitcast=True) -  base_float

    quantize_bits = quantize_float.to(tl.int32, bitcast=True)
    clip_quantize = clip_max_exponent(quantize_bits, max_exp, elem_mbits-2)
    quantize_float = clip_quantize.to(tl.float32, bitcast=True)

    tl.store(output_ptr + offsets, quantize_float, mask)

@triton.autotune(configs=get_cuda_autotune_config(),
  key=['M', 'N'] # the two above configs will be evaluated anytime
                 # the value of x_size changes
)
@triton.jit
def quantize_bfp_kernel_down (input_ptr, max_ptr, n,
                        output_ptr,
                        M, N,
                        symmetric, elem_mbits,
                        BLOCK_SIZE: tl.constexpr,
                        # **META
                        ):
    # BLOCK_SIZE = META['BLOCK_SIZE']
    
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    max_entry = tl.load(max_ptr + offsets, mask)
    max_entry_bits = max_entry.to(tl.int32, bitcast=True)
    
    a = tl.load(input_ptr + offsets, mask)
    conditon_tensor = ((a == -max_entry) & ((max_entry_bits >> 16 << 25) == 0xFE000000)) & (~symmetric)
    max_entry_bits = tl.where(conditon_tensor, (max_entry_bits >> 23) + 1 << 23, max_entry_bits)
    
    max_exp = max_entry_bits << 1 >> 24 << 23
    base_float = 6 * max_exp.to(tl.float32, bitcast=True)

    target_rebase = a + base_float
    target_bits = target_rebase.to(tl.uint32, bitcast=True)
    quantized = round_bitwise_down(target_bits, elem_mbits)
    quantize_float = quantized.to(tl.float32, bitcast=True) -  base_float

    quantize_bits = quantize_float.to(tl.uint32, bitcast=True)
    clip_quantize = clip_max_exponent(quantize_bits, max_exp, elem_mbits-2)
    quantize_float = clip_quantize.to(tl.float32, bitcast=True)

    tl.store(output_ptr + offsets, quantize_float, mask)

@triton.autotune(configs=get_cuda_autotune_config(),
  key=['M', 'N'] # the two above configs will be evaluated anytime
                 # the value of x_size changes
)
@triton.jit
def quantize_bfp_kernel_up (input_ptr, max_ptr, n,
                        output_ptr,
                        M, N,
                        symmetric, elem_mbits,
                        BLOCK_SIZE: tl.constexpr,
                        # **META
                        ):
    # BLOCK_SIZE = META['BLOCK_SIZE']
    
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    max_entry = tl.load(max_ptr + offsets, mask)
    max_entry_bits = max_entry.to(tl.int32, bitcast=True)
    
    a = tl.load(input_ptr + offsets, mask)
    conditon_tensor = ((a == -max_entry) & ((max_entry_bits >> 16 << 25) == 0xFE000000)) & (~symmetric)
    max_entry_bits = tl.where(conditon_tensor, (max_entry_bits >> 23) + 1 << 23, max_entry_bits)
    
    max_exp = max_entry_bits << 1 >> 24 << 23
    base_float = 6 * max_exp.to(tl.float32, bitcast=True)

    target_rebase = a + base_float
    target_bits = target_rebase.to(tl.uint32, bitcast=True)
    quantized = round_bitwise_up(target_bits, elem_mbits)
    quantize_float = quantized.to(tl.float32, bitcast=True) -  base_float

    quantize_bits = quantize_float.to(tl.uint32, bitcast=True)
    clip_quantize = clip_max_exponent(quantize_bits, max_exp, elem_mbits-2)
    quantize_float = clip_quantize.to(tl.float32, bitcast=True)

    tl.store(output_ptr + offsets, quantize_float, mask)

def quantize_bfp(in_tensor, scale_bits, ebits, mbits, max_norm, max_values, 
                symmetric, axis, block_size, rounding_mode="S"):
    out_tensor = torch.empty_like(in_tensor, device = in_tensor.device)
    ndim = in_tensor.dim()
    input_size = in_tensor.size()
    axis_size = input_size[axis]
    pre_axis_size=1
    for i in range(axis):
        pre_axis_size *= input_size[i]

    post_axis_size = 1
    for i in range(axis+1, ndim):
        post_axis_size *= input_size[i]

    total_size = pre_axis_size * axis_size * post_axis_size

    grid = get_grid(total_size)

    BLOCK_SIZE = block_size

    rounding_mode = ROUNDING_MODE_INT[rounding_mode]

    zero_tensor = torch.zeros_like(in_tensor)

    tensor_shape = list(in_tensor.shape)
    W = tensor_shape[0]
    H = tensor_shape[1]

    in_tensor = in_tensor.contiguous()
    zero_tensor = zero_tensor.contiguous()
    max_values = max_values.contiguous()
    out_tensor = out_tensor.contiguous()

    # call the triton kernel
    if rounding_mode == 0:
        quantize_bfp_kernel_up[grid](in_tensor, max_values, total_size,
                                out_tensor,
                                W, H,
                                symmetric, mbits,
                                # BLOCK_SIZE=BLOCK_SIZE,
                                )
        
    elif rounding_mode == 1:
        quantize_bfp_kernel_down[grid](in_tensor, max_values, total_size,
                                out_tensor,
                                W, H,
                                symmetric, mbits,
                                # BLOCK_SIZE=BLOCK_SIZE,
                                )
        
    elif rounding_mode == 2:
        quantize_bfp_kernel_nearest[grid](in_tensor, max_values, total_size,
                                out_tensor,
                                W, H,
                                symmetric, mbits,
                                # BLOCK_SIZE=BLOCK_SIZE,
                                )
    elif rounding_mode == 3:
        quantize_bfp_kernel_stochastic[grid](in_tensor, max_values, total_size,
                                out_tensor,
                                W, H,
                                symmetric, mbits,
                                # BLOCK_SIZE=BLOCK_SIZE,
                                )
        
    else:
        print("No such rounding mode present")
        return in_tensor
    
    return out_tensor
