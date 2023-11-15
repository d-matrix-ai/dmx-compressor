import numpy as np
import torch
import torch.nn.functional as F
from functools import wraps
from typing import List, Optional, Callable, Tuple
from types import SimpleNamespace
import math

try:
    from dmsimd import SIMDKernels as _K

except ModuleNotFoundError:

    class _K:
        @staticmethod
        def add(a, b):
            """
            Adds two numbers using the tieops add operation.

            Args:
                a (float): The first operand.
                b (float): The second operand.

            Returns:
                float: The result of a + b.
            """
            pass

        @staticmethod
        def tanh(x):
            """
            Compute the hyperbolic tangent of a number.

            Args:
                x (float): The input value.

            Returns:
                float: The hyperbolic tangent of x.
            """
            pass

        @staticmethod
        def sigmoid(x):
            """
            Compute the sigmoid function of a number.

            Args:
                x (float): The input value.

            Returns:
                float: The sigmoid of x.
            """
            pass

        @staticmethod
        def relu(x):
            """
            Compute the Rectified Linear Unit (ReLU) function of a number.

            Args:
                x (float): The input value.

            Returns:
                float: The ReLU of x.
            """
            pass

        @staticmethod
        def gelu(x):
            """
            Compute the Gaussian Error Linear Unit (GELU) function of a number.

            Args:
                x (float): The input value.

            Returns:
                float: The GELU of x.
            """
            return x

        @staticmethod
        def layernorm_reduction(x, norm=1):
            """
            Apply Layer Normalization to a tensor.

            Args:
                x (float): The input tensor.
                mean (float): The mean of the tensor.
                variance (float): The variance of the tensor.
                epsilon (float): A small constant to prevent division by zero (default 1e-6).

            Returns:
                float: The normalized tensor.
            """
            return x, x

        @staticmethod
        def layernorm_elementwise(
            x, mean, variance, weight, bias, norm=1, epsilon=1e-6
        ):
            """
            Apply Layer Normalization Elementwise operation to a tensor.
            Args:
                x (float): The input tensor.
                mean (float): The mean of the tensor.
                variance (float): The variance of the tensor.
                epsilon (float): A small constant to prevent division by zero (default 1e-6).

            Returns:
                float: The normalized tensor.
            """
            return x

        @staticmethod
        def softmax(input_array):
            """
            Calculate the softmax of an input array.

            Args:
                input_array (list or numpy.ndarray): The input array containing numeric values.

            Returns:
                list or numpy.ndarray: The softmax probabilities as a new array of the same shape as the input_array.
            """
            return input_array

        @staticmethod
        def silu(x):
            """
            Compute the Sigmoid-weighted Linear Unit (SiLU) function of a number.

            Args:
                x (float): The input value.

            Returns:
                float: The SiLU of x.
            """
            return x


__ALL__ = ["gelu", "layer_norm", "softmax", "hf_diffusers_timesteps"]


class K(SimpleNamespace):
    """
    Collection of wrapped SIMD kernels to interface with torch.Tensor
    """

    TILE_SIZE = 64
    NORM = 1

    def elemwise_kernel(func):
        @wraps(func)
        def wrapper(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            device, dtype, shape = x.device, x.dtype, x.shape
            x = x.cpu().numpy()
            x = np.array([func(_x) for _x in x])
            return torch.Tensor(x).to(dtype).to(device).view(shape)

        return wrapper

    gelu = elemwise_kernel(_K.gelu)
    # sin = elemwise_kernel(_K.sin)
    # cos = elemwise_kernel(_K.cos)

    def layernorm_reduction(
        x: torch.Tensor, norm: int = NORM
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = x.device, x.dtype
        sums, sumsq = _K.layernorm_reduction(x.cpu().numpy().flatten())
        return (
            torch.Tensor([sums]).to(dtype).to(device),
            torch.Tensor([sumsq]).to(dtype).to(device),
        )

    def layernorm_elementwise(
        x: torch.Tensor,
        m: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        cols: int,
        norm: int = NORM,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        x = x.cpu().numpy()
        m = m.cpu().numpy()
        v = v.cpu().numpy()
        g = g.cpu().numpy()
        b = b.cpu().numpy()
        x = _K.layernorm_elementwise(x, m, v, g, b, cols, norm, eps)
        return torch.Tensor(x).to(dtype).to(device)

    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        device, dtype = a.device, a.dtype
        return (
            torch.Tensor(_K.add(a.cpu().numpy(), b.cpu().numpy())).to(dtype).to(device)
        )

    def softmax(x: torch.Tensor) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        x = x.cpu().numpy()
        x = np.array([_K.softmax(_x) for _x in x])
        return torch.Tensor(x).to(dtype).to(device)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return K.gelu(x)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_shape = x.shape
    N = x_shape[dim]
    x = x.transpose(dim, -1).view(-1, N)
    x = K.softmax(x)
    return x.transpose(dim, -1).reshape(x_shape)


def _split_layer_norm_tensors(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    tile_size: int,
):
    input_shape = list(input.shape)
    if input_shape[-len(normalized_shape) :] != list(normalized_shape):
        raise ValueError(
            f"incompatible input shape {input_shape} and normalized_shape {normalized_shape}"
        )
    cols = np.prod(normalized_shape)
    input = input.view(-1, cols)
    nrows, ncols = input.shape[0], input.shape[1]
    if ncols % tile_size != 0:
        raise ValueError(
            f"reshaped layer_norm input {nrows}x{ncols} of not a multiple of tile size 1x{tile_size}"
        )
    input_tiles = [
        torch.split(_tr, tile_size, dim=1) for _tr in torch.split(input, 1, dim=0)
    ]
    if weight is not None:
        if weight.numel() % tile_size != 0:
            raise ValueError(
                f"layer_norm weight count {weight.numel()} of not a multiple of tile size {tile_size}"
            )
        weight = weight.view(-1)
        weight_tiles = torch.split(weight, tile_size)
    else:
        weight_tiles = [1.0] * (ncols // tile_size)
    if bias is not None:
        if bias.numel() % tile_size != 0:
            raise ValueError(
                f"layer_norm bias count {weight.numel()} of not a multiple of tile size {tile_size}"
            )
        bias = bias.view(-1)
        bias_tiles = torch.split(bias, tile_size)
    else:
        bias_tiles = [1.0] * (ncols // tile_size)
    return input_tiles, weight_tiles, bias_tiles, cols


def _bintree_allreduce(
    blocks: List[torch.Tensor],
    red_func: Callable = K.add,
):
    len_blocks = len(blocks)
    if len_blocks == 1:
        return blocks[0]
    else:
        head = blocks[: len_blocks // 2]
        tail = blocks[len_blocks // 2 :]
        for i in range(len(head)):
            tail[i] = red_func(head[i], tail[i])
        return _bintree_allreduce(tail)


def layer_norm(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    tile_size: int = K.TILE_SIZE,  # extra arg
    norm: int = K.NORM,  # extra arg
    **kwargs,
) -> torch.Tensor:
    input_tiles, weight_tiles, bias_tiles, cols = _split_layer_norm_tensors(
        input, normalized_shape, weight, bias, tile_size
    )
    output = []
    for input_tile_row in input_tiles:
        _sum, _sumsq = [], []
        for _input_tile in input_tile_row:
            _s, _sq = K.layernorm_reduction(_input_tile, norm)
            _sum.append(_s)
            _sumsq.append(_sq)
        _sum = _bintree_allreduce(_sum, red_func=K.add)
        _sumsq = _bintree_allreduce(_sumsq, red_func=K.add)
        output_tiles = []
        for _input_tile, _gamma_tile, _beta_tile in zip(
            input_tile_row, weight_tiles, bias_tiles
        ):
            output_tiles.append(
                K.layernorm_elementwise(
                    _input_tile, _sum, _sumsq, _gamma_tile, _beta_tile, cols, norm, eps
                )
            )
        output.append(torch.cat(output_tiles, dim=1))
    output = torch.cat(output, dim=0)
    return output.view_as(input)


def hf_diffusers_timesteps(
    timesteps: torch.Tensor,
    num_channels: int,
    half_dim: int,
    exponent: float,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    emb = timesteps[:, None].half() * exponent.half()[None, :]  # this is not allowed
    emb *= scale  # need kernels, annotate shape and dtype
    emb = torch.cat([K.sin(emb), K.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if num_channels % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb
