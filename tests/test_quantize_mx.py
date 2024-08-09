"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import pytest
import torch
import numpy as np
import sys

sys.path.insert(0, '../src')

from dmx.compressor.pt2bfp.quant.triton.common_lib import (
        check_diff,
        check_diff_quantize,
        all_encodings
)

from dmx.compressor.pt2bfp.quant.triton.mx_ops import _quantize_mx
from dmx.compressor import numerical

from dmx.compressor.numerical.format import Format

np.random.seed(0xd10)

MXFP8_E4M3K128_LD=Format.from_shorthand("MXFP8[E4M3]{128}")
MXFP8_E4M3K128_FD=Format.from_shorthand("MXFP8[E4M3]{128}")
MXFP8_E4M3K64_LD=Format.from_shorthand("MXFP8[E4M3]{64}")
MXFP8_E4M3K64_FD=Format.from_shorthand("MXFP8[E4M3]{64}")
MXFP8_E5M2K128_LD=Format.from_shorthand("MXFP8[E5M2]{128}")
MXFP8_E5M2K128_FD=Format.from_shorthand("MXFP8[E5M2]{128}")
MXFP8_E5M2K64_LD=Format.from_shorthand("MXFP8[E5M2]{64}")
MXFP8_E5M2K64_FD=Format.from_shorthand("MXFP8[E5M2]{64}")

DEVICE__CUSTOM_CUDA = [
    ("cuda", True),
]

ELEM_FMTS = [
    (MXFP8_E4M3K128_LD),
    (MXFP8_E5M2K128_LD),
]

'''
Code derived from https://github.com/microsoft/microxcaling/blob/b601f4e946fffd702ec8f2d41980b1788557d1b1/mx/tests/test_quantize_mx.py#L44
'''
@pytest.mark.parametrize("elem_format", ELEM_FMTS)
@pytest.mark.parametrize("round", ("N"))
@pytest.mark.parametrize("flush_fp32_subnorms", (False,True))
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_mx_encoding(elem_format, round,
                     flush_fp32_subnorms, device, custom_cuda):
    # print("elem_format", elem_format)
    # print("flush_fp32_subnorms", flush_fp32_subnorms)
    # print("device", device)
    scale_bits = elem_format.element_format.bit_precision
    block_size = elem_format.block_size
    x1 = torch.rand((1024, 128), device="cuda")
    x2 = x1.clone().detach().to(device)
    x3 = x1.clone().detach().to(device)
    # x1 = all_encodings(8, 9, device="cuda")
    # x2 = x1.clone().detach().to(device)
    
    y1 = _quantize_mx(x1, scale_bits, elem_format,
                  block_size=block_size,
                  axes=[-1],
                  round=round,
                  flush_fp32_subnorms=flush_fp32_subnorms,
                  custom_cuda=False)

    y2 = _quantize_mx(x2, scale_bits, elem_format,
                      block_size=block_size,
                      axes=[-1],
                      round=round,
                      flush_fp32_subnorms=flush_fp32_subnorms,
                      custom_cuda=custom_cuda)

    y3 = _quantize_mx(x3, scale_bits, elem_format,
                      block_size=block_size,
                      axes=[-1],
                      round=round,
                      flush_fp32_subnorms=flush_fp32_subnorms,
                      custom_cuda=custom_cuda,
                      check_mx=True) # if true run cuda kernel
    
    # check_diff_quantize(x1, y1, y2)
    # check_diff_quantize(x1, y1, y3)
    check_diff_quantize(x1, y3, y2)
