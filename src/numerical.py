from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize


__ALL__ = ["Same", "FixedPoint", "FloatingPoint", "BlockFloatingPoint", "CastTo"]


class Format:
    r"""
    This is an abstract class of tensor numerical format.
    Child classes to implement `cast()` method.
    """

    def __init__(self):
        pass

    def __str__(self) -> str:
        raise NotImplementedError

    def cast(self, *input: Any) -> None:
        raise NotImplementedError


class Same(Format):
    r"""
    This is a dummy numerical format whose `cast()` does not do anything but passing same input through.
    """

    def __init__(self):
        super().__init__()

    def cast(self, x):
        return x

    def __str__(self) -> str:
        return f"Dummy numerical format: no casting"

    def __repr__(self) -> str:
        return f"SAME"


class FixedPoint(Format):
    r"""
    This is a fixed point format simulated in FP32, using QPyTorch.
    """

    def __init__(
        self, precision, fraction, clamp=True, symmetric=True, rounding="nearest"
    ):
        super().__init__()
        # check validity of format configuration
        assert (
            1 <= precision <= 25
        ), f"highest integer precision simulated by FP32 is 25, got {precision}"
        # [TODO] check fraction validity, considering 8-bit exponent of FP32

        self.precision = precision
        self.fraction = fraction
        self.clamp = clamp
        self.symmetric = symmetric
        self.rounding = rounding

    def cast(self, x):
        return fixed_point_quantize(
            x,
            wl=self.precision,
            fl=self.fraction,
            clamp=self.clamp,
            symmetric=self.symmetric,
            rounding=self.rounding,
        )

    def __str__(self) -> str:
        return f"Simulated fixed point format: precision bits = {self.precision}, fraction bits = {self.fraction}, \ncasting behavior: symmetric = {self.symmetric}, clamp = {self.clamp}, rounding = {self.rounding}"

    def __repr__(self) -> str:
        return f"XP[{self.precision}]{self.fraction:+d}"


class FloatingPoint(Format):
    r"""
    This is a floating point format simulated in FP32, using QPyTorch.
    """

    def __init__(self, mantissa=23, exponent=8, rounding="nearest"):
        super().__init__()
        # check validity of format configuration
        assert (
            0 <= mantissa <= 23
        ), f"number of mantisa bits simulatable by FP32 is between 0 and 23, got{mantissa}"
        assert (
            0 < exponent <= 8
        ), f"number of exponent bits simulatable by FP32 is between 1 and 8, got {exponent}"

        self.mantissa = mantissa
        self.exponent = exponent
        self.rounding = rounding

    def cast(self, x):
        return float_quantize(
            x,
            man=self.mantissa,
            exp=self.exponent,
            rounding=self.rounding,
        )

    def __str__(self) -> str:
        return f"Simulated floating point format: mantissa bits = {self.mantissa}, exponent bits = {self.exponent}, \ncasting behavior: rounding = {self.rounding}"

    def __repr__(self) -> str:
        return f"FP[1|{self.exponent}|{self.mantissa}]"


class BlockFloatingPoint(Format):
    r"""
    This is a block floating point format simulated in FP32, using QPyTorch.
    """

    def __init__(self, precision, block_size=64, block_dim=-1, rounding="nearest"):
        super().__init__()
        # check validity of format configuration
        assert (
            1 <= precision <= 25
        ), f"highest integer precision simulated by FP32 is 25, got {precision}"
        assert block_size > 0, f"block size has to be positive, got {block_size}"

        self.precision = precision
        self.block_size = block_size
        self.block_dim = block_dim
        self.rounding = rounding

    def cast(self, x):
        # input of Linear: [B, ..., Cin], dim=-1
        # weight of Linear: [Cout, Cin], dim=-1
        # input of Conv1D: [B, Cin, L], dim=1
        # weight of Conv1D: [Cout, Cin//G, K], dim=1
        # input of Conv2D: [B, Cin, H, W], dim=1
        # weight of Conv2D: [Cout, Cin//G, K, K], dim=1

        x.transpose_(self.block_dim, -1)  # dim swap
        xshape = x.shape  # remember shape
        _x = torch.split(
            x.view(-1, xshape[-1]), self.block_size, dim=-1
        )  # slice to blocks
        x = torch.cat(
            [
                block_quantize(block, wl=self.precision, dim=0, rounding=self.rounding)
                for block in _x
            ],
            dim=self.block_dim,
        )  # quantize
        x = x.view(xshape).transpose_(self.block_dim, -1)  # recover shape
        return x

    def __str__(self) -> str:
        return f"Simulated fixed point format: precision bits = {self.precision}, fraction bits = {self.fraction}, \ncasting behavior: symmetric = {self.symmetric}, clamp = {self.clamp}, rounding = {self.rounding}"

    def __repr__(self) -> str:
        return f"BFP[{self.precision}|8]-{self.block_size}"


class CastToFormat(Function):
    r"""
    A simple STE backward function for numerical cast
    """

    @staticmethod
    def forward(ctx, x, fmt):
        return fmt.cast(x)

    @staticmethod
    def backward(ctx, g):
        return g, None


class CastTo(nn.Module):
    r"""
    Simulated numerical cast to a target format
    """

    def __init__(self, format=Same()):
        super().__init__()
        self.format = format

    def forward(self, x):
        return CastToFormat.apply(x, self.format) 

    def extra_repr(self):
        return f"format = {self.format.__repr__()}"


if __name__ == "__main__":
    pass
