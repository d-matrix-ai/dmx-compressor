'''
Replication in Triton of mltools BFP CUDA implementation
'''
import os
import torch
import numpy as np

from mltools.numerical.format import (Format, 
                                      ROUNDING_MODE)
from mltools.quant.quant_function import block_quantize
from mltools import numerical
from mltools.numerical import CastTo

from . import funcs

from .common import (
    FLOAT32_EXP_BIAS,
    FLOAT32_MIN_NORMAL,
    get_max_entry,
    ROUNDING_MODE_INT,
)
from .formats import (
        _get_format_params
    )
from .common import get_grid

import triton
import triton.language as tl



# -------------------------------------------------------------------------
# Helper funcs
# -------------------------------------------------------------------------
'''
Code derived from https://github.com/microsoft/microxcaling/blob/b601f4e946fffd702ec8f2d41980b1788557d1b1/mx/mx_ops.py#L95
'''
def _reshape_to_blocks(A, axes, block_size):
    if axes is None:
        raise Exception(
            "axes required in order to determine which "
            "dimension to apply block size to"
        )
    if block_size == 0:
        raise Exception("block_size == 0 in _reshape_to_blocks")

    # Fix axes to be positive and sort them
    axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
    assert all(x >= 0 for x in axes)
    axes = sorted(axes)

    # Add extra dimension for tiles
    for i in range(len(axes)):
        axes[i] += i  # Shift axes due to added dimensions
        A = torch.unsqueeze(A, dim=axes[i] + 1)

    # Pad to block_size
    orig_shape = A.size()
    pad = []
    for i in range(len(orig_shape)):
        pad += [0, 0]

    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if isinstance(pre_pad_size, torch.Tensor):
            pre_pad_size = int(pre_pad_size.value)
        # Don't pad if the axis is short enough to fit inside one tile
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True

    if do_padding:
        pad = list(reversed(pad))
        A = torch.nn.functional.pad(A, pad, mode="constant")

    def _reshape(shape, reshape_block_size):
        for axis in axes:
            # Reshape to tiles if axis length > reshape_block_size
            if shape[axis] >= reshape_block_size:
                assert shape[axis] % reshape_block_size == 0
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            # Otherwise preserve length and insert a 1 into the shape
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    # Reshape to tiles
    padded_shape = A.size()
    reshape = _reshape(list(padded_shape), block_size)

    A = A.view(reshape)
    return A, axes, orig_shape, padded_shape

'''
Code derived from https://github.com/microsoft/microxcaling/blob/b601f4e946fffd702ec8f2d41980b1788557d1b1/mx/mx_ops.py#L157
'''
def _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes):
    # Undo tile reshaping
    # print(A.values)
    A = A.view(padded_shape)
    # Undo padding
    if not list(padded_shape) == list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        A = A[slices]
    for axis in reversed(axes):
        # Remove extra dimension
        A = torch.squeeze(A, dim=axis + 1)
    return A

# -------------------------------------------------------------------------
# Main funcs
# -------------------------------------------------------------------------
def _quantize_bfp(
    A,
    scale_bits,
    elem_format,    # can be None for no quantization
    axes=None,
    block_size=1024,
    round="stochastic",
    symmetric = True,
    format_str="BFP[8|8]{1}(SN)",
    custom_cuda=False,
):
    """
    Function used for BFP* quantization
    """
    # Shortcut for no quantization
    if elem_format == None:
        return A

    assert(scale_bits > 0)

    # Make sure axes is a list of non-negative numbers
    axes = [axes] if type(axes) == int else axes
    axes = [x + A.ndim if x < 0 else x for x in axes]

    # Custom CUDA only supports limited rounding modes
    custom_cuda = custom_cuda and round in ROUNDING_MODE_INT
    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)


    # Perform tiling to the hardware vector size
    if block_size > 0:
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(
            A, axes, block_size
        )

    ####################
    # Quantize
    ####################
    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

    if custom_cuda:
        # Custom CUDA code only supports a single axis
        if shared_exp_axes is None:
            axis = 0
        else:
            assert len(shared_exp_axes) == 1
            axis = shared_exp_axes[0]

        # implement in triton
        ndim = A.ndim
        total_size = A.numel()
        max_values = torch.empty_like(A)

        A = A.contiguous()
        max_values = max_values.contiguous()

        # grid = get_grid(total_size)
        max_values_ =  get_max_entry(A, axis)
        
        # A_shape = list(A.shape)
        # ndim = A.ndim
        # for i in range(axis):
        #     A_shape[i] = 1
        # for i in range(axis+1, ndim):
        #     A_shape[i] = 1
        # max_values = max_values.repeat(A_shape)
        # max_values = max_values.to("cuda")

        A = A.contiguous()
        max_values_ = max_values_.contiguous()

        if A.device.type == "cuda":
            A = funcs.quantize_bfp_func(
                A, scale_bits, ebits, mbits, max_norm,
                max_values_, symmetric, axis, block_size,
                round)
        else:
            raise ValueError("Unrecognized device type %s" % A.device.type)
    else:
        # A = numerical.CastTo(format=format_str)(A)
        if shared_exp_axes is None:
            axis = 0
        else:
            assert len(shared_exp_axes) == 1
            axis = shared_exp_axes[0]
        cast = CastTo(
            format=format_str,
            group_size=block_size,
        )
        A = cast(A)
        # A = block_quantize(A, mbits, dim=axis, symmetric=symmetric, rounding=round)

    if block_size:
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A

