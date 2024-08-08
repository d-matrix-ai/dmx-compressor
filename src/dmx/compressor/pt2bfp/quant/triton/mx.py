import torch
import triton.language as tl

from mltools.numerical.format import Format

default_format = Format.from_shorthand("BFP[8|8]{64}(SN)")

import triton

from .common import get_grid, get_biased_exponent, get_shared_scale, round_mantissa
from .quantize import quantize_elemwise
from .common import ROUNDING_MODE_INT


'''
Code derived from https://github.com/microsoft/microxcaling/blob/b601f4e946fffd702ec8f2d41980b1788557d1b1/mx/cpp/mx.cuh#L16
'''
@triton.jit
def quantize_mx_kernel  (input_ptr, zero_ptr, max_ptr, n,
                        output_ptr,
                        scale_bits, elem_ebits, elem_mbits, emax, max_norm,
                        rounding_mode, flush_fp32_subnorms,
                        BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    max_values = tl.load(max_ptr + offsets, mask)

    shared_exp = tl.floor(tl.log2(max_values + (2 ** (-126)) * (max_values == 0).to(tl.float32)))
    
    ebits_tensor = tl.zeros_like(shared_exp) + elem_ebits
    ebits_tensor = ebits_tensor.to(tl.float32)
    emax_ebits = tl.exp2(ebits_tensor-1)-1
    # Restrict to [-emax, emax] range
    shared_exp = tl.where(shared_exp > emax_ebits, tl.zeros_like(shared_exp) + float("NaN"), shared_exp)
    shared_exp = tl.where(shared_exp < -emax_ebits, -emax_ebits, shared_exp)

    input_values = tl.load(input_ptr+offsets, mask)

    input_values = tl.where((shared_exp > -127) & (~flush_fp32_subnorms), input_values, tl.zeros_like(shared_exp))
    # if flush_fp32_subnorms:
    #     input_values = tl.where(shared_exp > -127, input_values, tl.zeros_like(shared_exp))

    shared_exp = shared_exp - emax

    scale_emax = tl.exp2(scale_bits.to(tl.float32)-1) - 1
    shared_exp = tl.where(shared_exp > scale_emax, tl.zeros_like(shared_exp) + float("NaN"), shared_exp)
    shared_exp = tl.where(shared_exp < -scale_emax, -scale_emax, shared_exp)

    # Scale down input before quantization
    scaled_input = tl.fdiv(input_values, tl.exp2(shared_exp))

    private_exp = tl.floor(tl.log2(tl.abs(scaled_input) + (scaled_input == 0).to(tl.float32)))

    # The minimum representable exponent for 8 exp bits is -126
    min_exp = -(tl.exp2(ebits_tensor - 1)) + 2
    private_exp = tl.clamp(private_exp, min=min_exp, max=emax)

    mbits_tensor = tl.zeros_like(scaled_input) + elem_mbits
    mbits_tensor = mbits_tensor.to(tl.float32)

    # Perfrom left shift
    out = tl.fdiv(scaled_input, tl.exp2(private_exp)) * tl.exp2(mbits_tensor - 2)
    # Perform round mantissa
    out = round_mantissa(out, elem_mbits, rounding_mode, False)
    # Perform right shift to undo scaling
    out = tl.fdiv(out, tl.exp2(mbits_tensor - 2)) * tl.exp2(private_exp)

    out = tl.clamp(out, min=-max_norm, max=max_norm)
    # sign = tl.where(out == tl.abs(out), tl.zeros_like(out)+1, tl.zeros_like(out) - 1)
    # out = tl.where(tl.abs(out) > max_norm, sign * float("Inf"), out)

    # Scale up after quantization
    out = out * tl.exp2(shared_exp)
    
    tl.store(output_ptr + offsets, out, mask)
    
def quantize_mx(in_tensor, scale_bits, ebits, mbits, emax, max_norm, max_values, 
                axis, block_size, flush_fp32_subnorms, rounding_mode="N"):
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
    quantize_mx_kernel[grid](in_tensor, zero_tensor, max_values, total_size,
                             out_tensor,
                             scale_bits, ebits, mbits, emax, max_norm,
                             rounding_mode, flush_fp32_subnorms,
                             BLOCK_SIZE=BLOCK_SIZE)
    
    return out_tensor
