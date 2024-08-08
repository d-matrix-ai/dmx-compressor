import torch
import triton

import triton.language as tl

from .common import get_biased_exponent, get_sign, get_trailing_mantissa, construct_float

FLOAT32_EXP_BIAS = 127
FLOAT32_TRAILING_MBITS = 23
FLOAT32_FULL_MBITS = (FLOAT32_TRAILING_MBITS + 1) # 24
FLOAT32_IMPLIED1 = (1 << FLOAT32_TRAILING_MBITS)

'''
Code derived from https://github.com/microsoft/microxcaling/blob/b601f4e946fffd702ec8f2d41980b1788557d1b1/mx/cpp/quantize.cuh#L15
'''
@triton.jit
def shift_right_round_mantissa(mantissa: tl.tensor, is_subnorm: tl.tensor,
                               exp_diff: tl.tensor, mbits,
                               rounding_mode, allow_overflow):
    mantissa = tl.where(is_subnorm, mantissa, mantissa + (1 << 23))
    fp32_sig_bits = tl.where(is_subnorm, tl.zeros_like(exp_diff)+23, tl.zeros_like(exp_diff)+24)
    # if not is_subnorm:
    #     mantissa = mantissa + (1 << 23) #FLOAT32_IMPLIED1
    #     fp32_sig_bits = 24
    # else:
    #     fp32_sig_bits = 23
    tbits = tl.zeros_like(exp_diff)
    mask = tl.zeros_like(exp_diff)
    tie = tl.zeros_like(exp_diff)
    even = tl.zeros_like(exp_diff)

    # 2 is the integer currently assigned to nearest rounding mode
    if rounding_mode == 2:
        tbits = exp_diff + (fp32_sig_bits - mbits) # number of bits that will be removed
        mask = (1 << (tbits - 1)) - 1
        tie = ~(mantissa & mask)
        mask = 1 << tbits
        even = ~(mantissa & mask)

    mantissa = mantissa >> exp_diff
    mantissa = mantissa >> (fp32_sig_bits - mbits - 1)
    
    if rounding_mode == 2:
        check_mantissa = ((allow_overflow | mantissa) != ((1 << (mbits+1)) - 1)) & (~(tie & even))
        mantissa = tl.where(check_mantissa, mantissa+1, mantissa)

    mantissa = mantissa >> 1

    return mantissa.to(tl.int32)
    
'''
Code derived from - 
https://github.com/microsoft/microxcaling/blob/b601f4e946fffd702ec8f2d41980b1788557d1b1/mx/cpp/quantize.cuh#L64
'''
@triton.jit
def shift_left_mantissa(mantissa: tl.tensor,  exp_diff:tl.tensor, is_subnorm: tl.tensor, mbits):
    fp32_sig_bits = tl.where(is_subnorm, tl.zeros_like(exp_diff)+23, tl.zeros_like(exp_diff)+24)
    # if is_subnorm:
    #     fp32_sig_bits = 23
    # else:
    #     fp32_sig_bits = 24

    mantissa = mantissa << (fp32_sig_bits - mbits + exp_diff)
    # Handle overflow - don't shift when subnorm overflows into a normal
    ones_tensor = tl.zeros_like(mantissa)+1
    zeros_tensor = tl.zeros_like(mantissa)
    overflow = tl.where(mantissa >= zeros_tensor + (1 << fp32_sig_bits), ones_tensor, zeros_tensor)
    mantissa = tl.where(overflow & (~is_subnorm), mantissa >> 1, mantissa)
    implied_tensor = tl.zeros_like(mantissa) + (1 << 23) #FLOAT32_IMPLIED1
    mantissa = mantissa & (implied_tensor - 1)
    return overflow

'''
Code derived from https://github.com/microsoft/microxcaling/blob/b601f4e946fffd702ec8f2d41980b1788557d1b1/mx/cpp/quantize.cuh#L88
'''
@triton.jit
def quantize_elemwise(input:tl.tensor, bits, exp_bits, max_norm, 
                      rounding_mode, saturate_normals, allow_denorm):
    biased_exp = get_biased_exponent(input)
    sign = get_sign(input)
    tmant = get_trailing_mantissa(input)
    
    # Mantissa bits to quantize to
    mbits = bits - 1
    is_int = (exp_bits == 0)
    
    # Integers can be treated as having exp bias of 1
    if is_int:
        new_bias = tl.zeros_like(biased_exp) + 1
    else:
        new_bias = tl.zeros_like(biased_exp) + ((1 << (exp_bits)) - 1)
    new_bias = new_bias.to(tl.int32)
    new_biased_exp = biased_exp - 127 + new_bias # biased_exp - FLOAT32_EXP_BIAS + new_bias
    
    # Skip denorms
    cond_skip_denorms = ((not is_int) and (not allow_denorm)) and (new_biased_exp < 1)

    # Use exp_diff to truncate additional bits for subnorms
    # mbits includes implicit 1, so when new_biased_exp==0
    # we want exp_diff = 1 to truncate away 1 bit
    exp_diff = tl.where(new_biased_exp <= 0, 1-new_biased_exp, tl.zeros_like(new_biased_exp))
    exp_diff = tl.where(exp_diff > 24, tl.zeros_like(exp_diff) + 24, exp_diff)

    biased_exp_check = biased_exp == 0
    tmant = shift_right_round_mantissa(tmant, biased_exp_check,
                                       exp_diff, mbits, rounding_mode,
                                       not is_int)
    
    overflow = shift_left_mantissa(tmant, exp_diff, biased_exp_check, mbits)

    biased_exp = tl.where(overflow, biased_exp+1, biased_exp)
    
    # reconstruct the float number
    output = construct_float(sign, biased_exp, tmant)
    neg_max_norm_tensor = tl.zeros_like(output) - max_norm
    max_norm_tensor = tl.zeros_like(output) + max_norm
    biased_exp_tensor = tl.zeros_like(biased_exp) + 0xFF
    tmant_tensor = tl.zeros_like(tmant)
    output = tl.where((tl.abs(output) >  max_norm_tensor) & (is_int or saturate_normals), 
                      tl.where(sign, neg_max_norm_tensor, max_norm_tensor),
                      construct_float(sign, biased_exp_tensor, tmant_tensor))

    # if mantissa is 0, return 0, added condition at the end
    output = tl.where(tmant == 0 & cond_skip_denorms, tl.zeros_like(output), output)
    return output
    