from typing import Any
from parse import parse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize


__ALL__ = [
    "Format",
    "BoundaryCastMixin",
    "Same",
    "FixedPoint",
    "FloatingPoint",
    "BlockFloatingPoint",
    "CastTo",
]


class Format:
    r"""
    This is an abstract class of tensor numerical format.
    Child classes to implement `cast()` and `from_shorthand()` method.
    """

    def __init__(self):
        pass

    def __str__(self) -> str:
        raise NotImplementedError

    def cast(self, *input: Any):
        raise NotImplementedError

    @staticmethod
    def from_shorthand(sh: str):
        if sh.startswith("SAME"):
            return Same.from_shorthand(sh)
        elif sh.startswith("XP"):
            return FixedPoint.from_shorthand(sh)
        elif sh.startswith("FP"):
            return FloatingPoint.from_shorthand(sh)
        elif sh.startswith("BFP"):
            return BlockFloatingPoint.from_shorthand(sh)
        else:
            raise ValueError(f"unrecognized format shorthand: {sh}")


class Same(Format):
    r"""
    This is a dummy numerical format whose `cast()` does not do anything but passing same input through.
    """

    def __init__(self):
        super().__init__()

    def cast(self, x):
        return x.clone()

    @classmethod
    def from_shorthand(cls, sh: str):
        return cls()

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
            1 <= precision <= 24
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

    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse(
            "XP[{precision:d},{fraction:d}]({clamp:l}{symmetric:l}{rounding:l})", sh
        )
        return cls(
            precision=conf["precision"],
            fraction=conf["fraction"],
            clamp=conf["clamp"] == "C",
            symmetric=conf["symmetric"] == "S",
            rounding="stochastic" if conf["rounding"] == "S" else "nearest",
        )

    def __str__(self) -> str:
        return f"Simulated fixed point format: precision bits = {self.precision}, fraction bits = {self.fraction}, \ncasting behavior: symmetric = {self.symmetric}, clamp = {self.clamp}, rounding = {self.rounding}"

    def __repr__(self) -> str:
        return f"XP[{self.precision}{self.fraction:+d}]({'C' if self.clamp else 'U'}{'S' if self.symmetric else 'A'}{'S' if self.rounding=='stochastic' else 'N'})"


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
        return (
            x
            if self.mantissa == 23 and self.exponent == 8
            else float_quantize(
                x,
                man=self.mantissa,
                exp=self.exponent,
                rounding=self.rounding,
            )
        )

    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse("FP[1|{exponent:d}|{mantissa:d}]({rounding:l})", sh)
        return cls(
            mantissa=conf["mantissa"],
            exponent=conf["exponent"],
            rounding="stochastic" if conf["rounding"] == "S" else "nearest",
        )

    def __str__(self) -> str:
        return f"Simulated floating point format: mantissa bits = {self.mantissa}, exponent bits = {self.exponent}, \ncasting behavior: rounding = {self.rounding}"

    def __repr__(self) -> str:
        return f"FP[1|{self.exponent}|{self.mantissa}]({'S' if self.rounding=='stochastic' else 'N'})"


class BlockFloatingPoint(Format):
    r"""
    This is a block floating point format simulated in FP32, using QPyTorch.
    """

    def __init__(self, precision=8, block_size=64, block_dim=-1, rounding="nearest"):
        super().__init__()
        # check validity of format configuration
        assert (
            2 <= precision <= 25
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
        # TODO: modify qtorch kernels do the following at C++ level
        _x = x.transpose(self.block_dim, -1)  # dim swap
        xshape = _x.shape  # remember shape
        _xs = torch.split(
            _x.reshape((-1, xshape[-1])), self.block_size, dim=-1
        )  # slice to blocks
        _x = torch.cat(
            [
                block_quantize(block, wl=self.precision, dim=0, rounding=self.rounding)
                for block in _xs
            ],
            dim=self.block_dim,
        )  # quantize
        _x = _x.reshape(xshape).transpose_(self.block_dim, -1)  # recover shape
        return _x

    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse(
            "BFP[{precision:d}|8]{{{block_size:d},{block_dim:d}}}({rounding:l})", sh
        )
        return cls(
            precision=conf["precision"],
            block_size=conf["block_size"],
            block_dim=conf["block_dim"],
            rounding="stochastic" if conf["rounding"] == "S" else "nearest",
        )

    def __str__(self) -> str:
        return f"Simulated block floating point format: precision bits = {self.precision}, block size = {self.block_size}, block dimension = {self.block_dim}\ncasting behavior: rounding = {self.rounding}"

    def __repr__(self) -> str:
        return f"BFP[{self.precision}|8]{{{self.block_size},{self.block_dim}}}({'S' if self.rounding=='stochastic' else 'N'})"


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

    def __init__(self, format="SAME", dump_to=None):
        super().__init__()
        if not isinstance(format, Format):
            format = Format.from_shorthand(format)
        self.format = format
        self.dump_to = dump_to

    def forward(self, x):
        x = CastToFormat.apply(x, self.format) if x is not None else None
        if self.dump_to is not None:
            pass
        return x

    def extra_repr(self):
        return f"format = {self.format.__repr__()}"


class BoundaryCastMixin:
    r"""
    Mixin for modules with boundary casting
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.init_casts()

    def init_casts(self):
        # dynamic i/o casts
        self.input_cast = CastTo()
        self.output_cast = CastTo()
        # dynamic intermediate casts
        if (
            type(self)
            in (
                nn.Linear,
                nn.Bilinear,
                nn.EmbeddingBag,
            )
            or isinstance(self, nn.modules.conv._ConvNd)
        ):
            self.accum_cast = CastTo()
        else:
            self.accum_cast = None
        # static paramter casts
        pnames = [n for n, _ in self.named_parameters()]
        self.weight_cast = CastTo() if "weight" in pnames else None
        self.bias_cast = CastTo() if "bias" in pnames else None


if __name__ == "__main__":
    pass
