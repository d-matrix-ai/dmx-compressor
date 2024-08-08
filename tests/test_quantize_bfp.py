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

from dmx.compressor.pt2bfp.quant.triton.bfp_ops import _quantize_bfp
from dmx.compressor import numerical

from dmx.compressor.numerical.format import Format
RANDOM_SEED = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)

def test_castto_bfp16_1(bfp_format_str = "BFP[8|8]{64}(SU)"):
    n = 1024
    x = torch.randn((128, n), dtype=torch.float32).to(device)
    x *= 0.5 / x.abs().max()
    x += 1.0
    elem_format_str = bfp_format_str
    elem_format = Format.from_shorthand(elem_format_str)
    rounding = elem_format.rounding
    symmetric = elem_format.symmetric
    block_size = elem_format.block_size

    _x_cuda = _quantize_bfp(x, 16, elem_format,
                      axes=[-1],
                      block_size=block_size,
                      round=rounding,
                      symmetric=symmetric,
                      format_str=elem_format_str,
                      custom_cuda=False)
    

    _x_triton = _quantize_bfp(x, 16, elem_format,
                      axes=[-1],
                      block_size=block_size,
                      round=rounding,
                      symmetric=symmetric,
                      format_str=elem_format_str,
                      custom_cuda=True)
    print(x)
    print(_x_cuda)
    print(_x_triton)
    check_diff_quantize(x, _x_cuda, _x_triton)
    
test_castto_bfp16_1()