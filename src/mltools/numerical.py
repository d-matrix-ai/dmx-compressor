from typing import Any
from parse import parse
from bidict import bidict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from .quant import fixed_point_quantize, block_quantize, float_quantize

import numpy as np
from numerics.unary_functions import convert_FP32_to_FPsmall
from numerics import Data


__ALL__ = [
    "Format",
    "NumericalCastMixin",
    "Same",
    "FixedPoint",
    "FloatingPoint",
    "BlockFloatingPoint",
    "ScaledBlockFloatingPoint",
    "CastTo",
]


ROUNDING_MODE = bidict(
    {
        "U": "up",
        "D": "down",
        "N": "nearest",
        "S": "stochastic",
    }
)


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
        elif sh.startswith("SBFP"):
            return ScaledBlockFloatingPoint.from_shorthand(sh)
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
            "XP[{precision:d},{fraction:d}]({clamp:w}{symmetric:w}{rounding:w})", sh
        )
        return cls(
            precision=conf["precision"],
            fraction=conf["fraction"],
            clamp=conf["clamp"] == "C",
            symmetric=conf["symmetric"] == "S",
            rounding=ROUNDING_MODE[conf["rounding"]],
        )

    def __str__(self) -> str:
        return f"Simulated fixed point format: precision bits = {self.precision}, fraction bits = {self.fraction}, \ncasting behavior: symmetric = {self.symmetric}, clamp = {self.clamp}, rounding = {self.rounding}"

    def __repr__(self) -> str:
        return f"XP[{self.precision},{'0' if self.fraction==0 else f'{self.fraction:+d}'}]({'C' if self.clamp else 'U'}{'S' if self.symmetric else 'A'}{ROUNDING_MODE.inverse[self.rounding]})"


class FloatingPoint(Format):
    r"""
    This is a floating point format simulated in FP32, using QPyTorch.
    """

    def __init__(
        self,
        mantissa=23,
        exponent=8,
        bias=None,
        flush_subnormal=True,
        rounding="nearest",
    ):
        super().__init__()
        # check validity of format configuration
        assert (
            0 <= mantissa <= 23
        ), f"number of mantisa bits simulatable by FP32 is between 0 and 23, got{mantissa}"
        assert (
            0 < exponent <= 8
        ), f"number of exponent bits simulatable by FP32 is between 1 and 8, got {exponent}"
        _bias_min = 127 if exponent == 8 else -128 + 2**exponent
        assert (
            _bias_min <= bias <= 127
        ), f"exponent bias simulatable by FP32 for {exponent}-bit exponent is constrained between {_bias_min} and 127, got {bias}"

        self.mantissa = mantissa
        self.exponent = exponent
        self.bias = bias if bias is not None else 2 ** (exponent - 1) - 1
        self.flush_subnormal = flush_subnormal
        self.rounding = rounding

    def cast(self, x):
        return (
            x
            if self.mantissa == 23
            and self.exponent == 8
            and self.bias == 127
            and not self.flush_subnormal
            and self.rounding == "nearest"
            else float_quantize(
                x,
                man=self.mantissa,
                exp=self.exponent,
                bias=self.bias,
                flush_subnormal=self.flush_subnormal,
                rounding=self.rounding,
            )
        )

    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse(
            "FP[1|{exponent:d}|{mantissa:d},{bias:d}]({flush_subnormal:w}{rounding:l})",
            sh,
        )
        return cls(
            mantissa=conf["mantissa"],
            exponent=conf["exponent"],
            bias=conf["bias"],
            flush_subnormal=conf["flush_subnormal"] == "F",
            rounding=ROUNDING_MODE[conf["rounding"]],
        )

    def __str__(self) -> str:
        return f"Simulated floating point format: mantissa bits = {self.mantissa}, exponent bits = {self.exponent}, exponent bias = {self.bias}, \ncasting behavior: flush subnormal = {self.flush_subnormal}, rounding = {self.rounding}"

    def __repr__(self) -> str:
        return f"FP[1|{self.exponent}|{self.mantissa},{self.bias}]({'F' if self.flush_subnormal else '_'}{ROUNDING_MODE.inverse[self.rounding]})"


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
        self.bfp_id = -1

    def cast(self, x: torch.Tensor):
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
            rounding=ROUNDING_MODE[conf["rounding"]],
        )

    def __str__(self) -> str:
        return f"Simulated block floating point format: precision bits = {self.precision}, block size = {self.block_size}, block dimension = {self.block_dim}\ncasting behavior: rounding = {self.rounding}"

    def __repr__(self) -> str:
        return f"BFP[{self.precision}|8]{{{self.block_size},{self.block_dim}}}({ROUNDING_MODE.inverse[self.rounding]})"


class ScaledBlockFloatingPoint(Format):
    r"""
    This is a scaled block floating point tensor format.
    """

    def __init__(
        self,
        block_format: FixedPoint,
        scaler_format: FloatingPoint,
        block_size=64,
        block_dim=-1,
    ):
        super().__init__()
        # check validity of format configuration
        assert isinstance(
            block_format, FixedPoint
        ), "block format needs to be fixed point"
        assert isinstance(
            scaler_format, FloatingPoint
        ), "scaler format needs to be floating point"
        assert block_format.fraction == 0, "block format needs to have zero fraction"
        assert block_format.symmetric, "block format needs to have symmetric range"
        assert block_size > 0, f"block size has to be positive, got {block_size}"

        self.block_format = block_format
        self.scaler_format = scaler_format
        self.block_size = block_size
        self.block_dim = block_dim

        self.man_scaling = (
            2 ** (self.block_format.precision - 1) - 1
        )  # largest mantissa abs
        self.get_chunk_max = lambda chunk: torch.max(
            torch.abs(chunk), dim=-1, keepdim=True
        )[0]

    def cast(self, x: torch.Tensor) -> torch.Tensor:
        _x = x.transpose(self.block_dim, -1)  # dim swap
        xshape = _x.shape  # remember shape
        _xs = torch.split(
            _x.reshape((-1, xshape[-1])), self.block_size, dim=-1
        )  # slice to chunks
        _xms = [
            self.get_chunk_max(chunk) / self.man_scaling for chunk in _xs
        ]  # max of blocks in each chunk
        _x = torch.cat(
            [
                self.block_format.cast(chunk / chunk_max)
                * self.scaler_format.cast(chunk_max)
                for chunk, chunk_max in zip(_xs, _xms)
            ],
            dim=self.block_dim,
        )  # quantize
        _x = _x.reshape(xshape).transpose_(self.block_dim, -1)  # recover shape
        return _x

    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse(
            "SBFP<{block_format_sh}><{scaler_format_sh}>{{{block_size:d},{block_dim:d}}}",
            sh,
        )
        return cls(
            block_format=FixedPoint.from_shorthand(conf["block_format_sh"]),
            scaler_format=FloatingPoint.from_shorthand(conf["scaler_format_sh"]),
            block_size=conf["block_size"],
            block_dim=conf["block_dim"],
        )

    def __str__(self) -> str:
        return f"Simulated scaled block floating point format: block format = {self.block_format}, scaler format = {self.scaler_format},\n block size = {self.block_size}, block dimension = {self.block_dim}"

    def __repr__(self) -> str:
        return f"SBFP<{repr(self.block_format)}><{repr(self.scaler_format)}>{{{self.block_size},{self.block_dim}}}"


class CastToFormat(Function):
    r"""
    A simple STE backward function for numerical cast
    """

    @staticmethod
    def forward(ctx, x, fmt):
        ctx.set_materialize_grads(False)
        return fmt.cast(x)

    @staticmethod
    def backward(ctx, g):
        return g, None

    @staticmethod
    def symbolic(
        g: torch._C.Graph, input: torch._C.Value, fmt: torch._C.Value
    ) -> torch._C.Value:
        if isinstance(fmt, Same):
            return g.op("Identity", input)
        elif isinstance(fmt, BlockFloatingPoint):

            # TODO with dtype for torch > 1.11
            return g.op(
                "com.microsoft::DequantizeBFP",
                *g.op(
                    "com.microsoft::QuantizeBFP",
                    input,
                    bfp_type_i=torch.onnx.symbolic_helper._parse_arg(fmt.bfp_id, "i"),
                    outputs=3,
                ),
                bfp_type_i=torch.onnx.symbolic_helper._parse_arg(fmt.bfp_id, "i"),
                dtype_i=1,
                outputs=1,
            )
        else:
            return None


class CastTo(nn.Module):
    r"""
    Simulated numerical cast to a target format
    """

    def __init__(self, format="SAME", dump_to=None):
        super().__init__()
        self.set_format(format)
        self.dump_to = dump_to

    def set_format(self, format):
        if not isinstance(format, Format):
            format = Format.from_shorthand(format)
        self.format = format

    def forward(self, x):
        x = CastToFormat.apply(x, self.format) if x is not None else None
        if self.dump_to is not None:
            pass
        return x

    def extra_repr(self):
        return f"format = {self.format.__repr__()}"


class NumericalCastMixin:
    r"""
    Mixin for modules with boundary casting
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.init_casts()

    def init_casts(self):
        # dynamic i/o casts
        self.input_cast = CastTo()  # if isinstance(self, CorsairModule) else None
        self.output_cast = CastTo()  # if isinstance(self, CorsairModule) else None
        # dynamic intermediate casts
        if isinstance(
            self,
            (
                nn.Linear,
                nn.Bilinear,
                nn.EmbeddingBag,
                nn.modules.conv._ConvNd,
            ),
        ):
            self.accum_cast = CastTo()
        else:
            self.accum_cast = None
        # static paramter casts
        pnames = [n for n, _ in self.named_parameters()]
        self.weight_cast = CastTo() if "weight" in pnames else None
        self.bias_cast = CastTo() if "bias" in pnames else None

    @property
    def input_format(self):
        return repr(self.input_cast.format)

    @property
    def output_format(self):
        return repr(self.output_cast.format)

    @property
    def accum_format(self):
        return repr(self.accum_cast.format) if self.accum_cast is not None else None

    @property
    def weight_format(self):
        return repr(self.weight_cast.format) if self.weight_cast is not None else None

    @property
    def bias_format(self):
        return repr(self.bias_cast.format) if self.bias_cast is not None else None
