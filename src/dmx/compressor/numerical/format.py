from abc import ABC
from typing import Any, Optional
from parse import parse
from bidict import bidict
import torch

try:
    from ..quant import fixed_point_quantize, block_quantize, float_quantize
except ImportError as error:
    print("Error importing Block Quantize CUDA kernels")
from .onnx import BFPTypeEnum

try:
    from numerics import (
        determine_sbfp_scaler_exponent_bias_from_tensor_values,
    )

    NUMERICS_UTILS_AVAILABLE = True
except (ModuleNotFoundError,ImportError):
    NUMERICS_UTILS_AVAILABLE = False


ROUNDING_MODE = bidict(
    {
        "U": "up",
        "D": "down",
        "N": "nearest",
        "S": "stochastic",
    }
)


class Format(ABC):
    r"""
    This is an abstract class of tensor numerical format.
    Child classes to implement `cast()` and `from_shorthand()` method.
    """

    blocked: bool
    bfp_id: Optional[int] = None

    def __init__(self):
        pass

    def __str__(self) -> str:
        raise NotImplementedError

    def cast(self, *input: Any):
        raise NotImplementedError

    @property
    def bytes_per_elem(self) -> Optional[float]:
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
        elif sh.startswith("MXFP"):
            return MXFP.from_shorthand(sh)
        elif sh.startswith("MXINT"):
            return MXINT.from_shorthand(sh)
        else:
            raise ValueError(f"unrecognized format shorthand: {sh}")

    @property
    def bit_precision(self) -> Optional[float]:
        raise NotImplementedError


class Same(Format):
    r"""
    This is a dummy numerical format whose `cast()` does not do anything but passing same input through.
    """

    blocked = False

    def __init__(self):
        super().__init__()

    def cast(self, x, *args):
        return x.clone()

    @property
    def bytes_per_elem(self) -> None:
        return None

    @classmethod
    def from_shorthand(cls, sh: str):
        return cls()

    @property
    def bit_precision(self) -> Optional[float]:
        return None

    def __str__(self) -> str:
        return "Dummy numerical format: no casting"

    def __repr__(self) -> str:
        return "SAME"


class FixedPoint(Format):
    r"""
    This is a fixed point format simulated in FP32, using QPyTorch.
    """

    blocked = False

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

    def cast(self, x, *args):
        return fixed_point_quantize(
            x,
            wl=self.precision,
            fl=self.fraction,
            clamp=self.clamp,
            symmetric=self.symmetric,
            rounding=self.rounding,
        )

    @property
    def bytes_per_elem(self) -> float:
        return self.precision / 8.0

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

    @property
    def bit_precision(self) -> float:
        return float(self.precision)

    def __str__(self) -> str:
        return f"Simulated fixed point format: precision bits = {self.precision}, fraction bits = {self.fraction}, \ncasting behavior: symmetric = {self.symmetric}, clamp = {self.clamp}, rounding = {self.rounding}"

    def __repr__(self) -> str:
        return f"XP[{self.precision},{'0' if self.fraction==0 else f'{self.fraction:+d}'}]({'C' if self.clamp else '_'}{'S' if self.symmetric else '_'}{ROUNDING_MODE.inverse[self.rounding]})"


class FloatingPoint(Format):
    r"""
    This is a floating point format simulated in FP32, using QPyTorch.
    """

    blocked = False

    def __init__(
        self,
        mantissa=23,
        exponent=8,
        bias=None,
        flush_subnormal=True,
        unsigned=False,
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
        self.unsigned = unsigned
        self.rounding = rounding

    def cast(self, x, *args):
        x = (
            x
            if (x.dtype == torch.float32 and repr(self) == "FP[1|8|23,127](_N)")
            or (x.dtype == torch.float16 and repr(self) == "FP[1|5|10,15](_N)")
            else float_quantize(
                x.float(),
                man=self.mantissa,
                exp=self.exponent,
                bias=self.bias,
                flush_subnormal=self.flush_subnormal,
                rounding=self.rounding,
            )
        )
        # subnormal flushing for float16
        if repr(self) == "FP[1|5|10,15](FN)":
            smallest_normal = torch.finfo(torch.float16).smallest_normal
            subnormal_threshold = torch.tensor(
                smallest_normal, dtype=torch.float16, device=x.device
            )
            x = torch.where(
                x.abs() < subnormal_threshold,
                torch.tensor(0.0, dtype=torch.float16),
                x,
            )
        return x.abs() if self.unsigned else x

    @property
    def largest_representable_power_of_two(self):
        return 2 ** (2 ** (self.exponent - 1))

    @property
    def bytes_per_elem(self) -> float:
        return (self.mantissa + self.exponent + 1) / 8.0

    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse(
            "FP[{sign:d}|{exponent:d}|{mantissa:d},{bias:d}]({flush_subnormal:w}{rounding:l})",
            sh,
        )
        return cls(
            mantissa=conf["mantissa"],
            exponent=conf["exponent"],
            bias=conf["bias"],
            flush_subnormal=conf["flush_subnormal"] == "F",
            unsigned=conf["sign"] == 0,
            rounding=ROUNDING_MODE[conf["rounding"]],
        )

    @property
    def bit_precision(self) -> float:
        return float(
            self.mantissa + self.exponent
            if self.unsigned
            else 1 + self.mantissa + self.exponent
        )

    def __str__(self) -> str:
        return f"Simulated floating point format: mantissa bits = {self.mantissa}, exponent bits = {self.exponent}, exponent bias = {self.bias}, unsigned = {self.unsigned}, \ncasting behavior: flush subnormal = {self.flush_subnormal}, rounding = {self.rounding}"

    def __repr__(self) -> str:
        return f"FP[{'0' if self.unsigned else '1'}|{self.exponent}|{self.mantissa},{self.bias}]({'F' if self.flush_subnormal else '_'}{ROUNDING_MODE.inverse[self.rounding]})"


class BlockFloatingPoint(Format):
    r"""
    This is a block floating point format simulated in FP32, using QPyTorch.
    """

    blocked = True

    def __init__(
        self,
        precision=8,
        block_size=64,
        symmetric=True,
        rounding="nearest",
    ):
        super().__init__()
        # check validity of format configuration
        assert (
            2 <= precision <= 25
        ), f"highest integer precision simulated by FP32 is 25, got {precision}"
        assert block_size > 0, f"block size has to be positive, got {block_size}"

        self.precision = precision
        self.block_size = block_size
        self.symmetric = symmetric
        self.rounding = rounding

    @property
    def bfp_id(self):
        name = f"DMX_BFP_{self.precision+8}{'' if self.symmetric else 'A'}_{self.block_size}"
        return BFPTypeEnum[name].value

    def cast(self, x: torch.Tensor, block_dim: int):
        # input of Linear: [B, ..., Cin], dim=-1
        # weight of Linear: [Cout, Cin], dim=-1
        # input of Conv1D: [B, Cin, L], dim=1
        # weight of Conv1D: [Cout, Cin//G, K], dim=1
        # input of Conv2D: [B, Cin, H, W], dim=1
        # weight of Conv2D: [Cout, Cin//G, K, K], dim=1
        # TODO: modify qtorch kernels do the following at C++ level
        if self.block_size == 1:  # borrowing float_quantize for block_size==1
            _x = float_quantize(
                x.float(),
                man=self.precision - 2,  # 1 for sign and 1 for implicit bit
                exp=8,
                bias=127,
                flush_subnormal=False,
                rounding=self.rounding,
            )
        else:
            _x = x.float().transpose(block_dim, -1)  # dim swap
            xshape = _x.shape  # remember shape
            _xs = torch.split(
                _x.reshape((-1, xshape[-1])), self.block_size, dim=-1
            )  # slice to blocks
            _x =  [
                    block_quantize(
                        block,
                        wl=self.precision,
                        dim=0,
                        symmetric=True, #Force symmetric and postprocess to make asymmetric
                        rounding=self.rounding,
                    )
                    for block in _xs
                ]
            if not self.symmetric:
                _x = [self.make_mantissa_asymmetric(bfp16_block,fp32_block,self.precision) \
                      for (bfp16_block,fp32_block) in zip(_x,_xs)]
            _x = torch.cat(_x,dim = -1)
            _x = _x.reshape(xshape).transpose_(block_dim, -1)  # recover shape
            
        return _x

    @property
    def bytes_per_elem(self) -> float:
        return (self.precision + 8.0 / self.block_size) / 8.0

    @staticmethod
    def make_mantissa_asymmetric(dmx_result,fp32_inp,n_mantissa_bits = 8):
        man,exp = torch.frexp(dmx_result)
        exp[torch.logical_and(exp == 0,man == 0)] = -200 #A number that cannot be max exponent
        max_exp = exp.max(-1,keepdims = True)[0] - n_mantissa_bits + 1        
        exp_diff = exp - max_exp 
        int_man = (man * torch.pow(2.0,exp_diff)).int()

        #Check all locations that have a mantissa == -127, can we reduce the
        #quantization error if we change -127 to -128?. If we can,
        #then make the change. Make the change even if the quantization error stays the same
        #as we break the tie towards the even mantissa (-128)
        edge_locs = (int_man == -127).nonzero()
        if len(edge_locs) == 0:
            return dmx_result
        old_quant_error= dmx_result[edge_locs[:,0],edge_locs[:,1]] - fp32_inp[edge_locs[:,0],edge_locs[:,1]]
        candidate_quant_error = old_quant_error - torch.pow(2.0,max_exp[:,0][edge_locs[:,0]])
        subtract_1_locs = edge_locs[candidate_quant_error.abs() <= old_quant_error.abs()]

        int_man[subtract_1_locs[:,0],subtract_1_locs[:,1]] -= 1
        new_result = torch.ldexp(int_man,max_exp)

        return new_result
    
    
    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse(
            "BFP[{precision:d}|8]{{{block_size:d}}}({symmetric:w}{rounding:l})",
            sh,
        )

        return cls(
            precision=conf["precision"],
            block_size=conf["block_size"],
            symmetric=conf["symmetric"] == "S",
            rounding=ROUNDING_MODE[conf["rounding"]],
        )

    @property
    def bit_precision(self) -> float:
        return self.precision + 8.0 / self.block_size

    def __str__(self) -> str:
        return f"Simulated block floating point format: precision bits = {self.precision}, block size = {self.block_size}\ncasting behavior: symmetric = {self.symmetric}, rounding = {self.rounding}"

    def __repr__(self) -> str:
        return f"BFP[{self.precision}|8]{{{self.block_size}}}({'S' if self.symmetric else '_'}{ROUNDING_MODE.inverse[self.rounding]})"


class ScaledBlockFloatingPoint(Format):
    r"""
    This is a scaled block floating point tensor format.
    """

    blocked = True

    def __init__(
        self,
        block_format: FixedPoint,
        scaler_format: FloatingPoint,
        block_size=64,
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

        self.man_scaling = (
            2 ** (self.block_format.precision - 1) - 1
        )  # largest mantissa abs
        self.get_chunk_max = lambda chunk: torch.max(
            torch.abs(chunk), dim=-1, keepdim=True
        )[0]

        self.scaler_format_exponent_bias_determined = False

    def determine_scaler_exponent_bias_from(self, x: torch.Tensor) -> None:
        r"""
        Determines scaler_format.bias based on the value of the quantized tensor
        as a side-effect
        """
        if NUMERICS_UTILS_AVAILABLE:
            self.scaler_format.bias = (
                determine_sbfp_scaler_exponent_bias_from_tensor_values(x)
            )

    @property
    def bfp_id(self):
        name = f"DMX_SBFP_{self.block_format.precision+8}_{self.block_size}_{self.scaler_format.bias}"
        return BFPTypeEnum[name].value

    def cast(self, x: torch.Tensor, block_dim: int) -> torch.Tensor:
        if not self.scaler_format_exponent_bias_determined:
            self.determine_scaler_exponent_bias_from(x)
            self.scaler_format_exponent_bias_determined = True
        _x = x.float().transpose(block_dim, -1)  # dim swap
        xshape = _x.shape  # remember shape
        _xs = torch.split(
            _x.reshape((-1, xshape[-1])), self.block_size, dim=-1
        )  # slice to chunks
        _xms = [
            self.get_chunk_max(chunk) / self.man_scaling for chunk in _xs
        ]  # max of blocks in each chunk
        _x = torch.cat(
            [
                torch.where(
                    chunk_max > 0.0,
                    self.block_format.cast(chunk / chunk_max)
                    * self.scaler_format.cast(chunk_max),
                    chunk,
                )
                for chunk, chunk_max in zip(_xs, _xms)
            ],
            dim=-1,
        )  # quantize

        _x = _x.reshape(xshape).transpose_(block_dim, -1)  # recover shape
        return _x

    @property
    def bytes_per_elem(self) -> float:
        return (
            self.block_format.bytes_per_elem
            + self.scaler_format.bytes_per_elem / self.block_size
        )

    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse(
            "SBFP<{block_format_sh}><{scaler_format_sh}>{{{block_size:d}}}",
            sh,
        )
        return cls(
            block_format=FixedPoint.from_shorthand(conf["block_format_sh"]),
            scaler_format=FloatingPoint.from_shorthand(conf["scaler_format_sh"]),
            block_size=conf["block_size"],
        )

    @property
    def bit_precision(self) -> float:
        return (
            self.block_format.bit_precision
            + self.scaler_format.bit_precision / self.block_size
        )

    def __str__(self) -> str:
        return f"Simulated scaled block floating point format: block format = {self.block_format}, scaler format = {self.scaler_format},\n block size = {self.block_size}"

    def __repr__(self) -> str:
        return f"SBFP<{repr(self.block_format)}><{repr(self.scaler_format)}>{{{self.block_size}}}"


class MXFP(Format):
    r"""
    This is a MXFP tensor format.
    """

    blocked = True

    def __init__(
        self,
        element_format: FloatingPoint,
        block_size=32,
    ):
        super().__init__()
        # check validity of format configuration
        assert isinstance(
            element_format, FloatingPoint
        ), "block format needs to be floating point"
        assert block_size > 0, f"block size has to be positive, got {block_size}"
        self.element_format = element_format
        self.scaler_format = FloatingPoint(
            mantissa=0,
            exponent=8,
            bias=127,
            unsigned=True,
        )
        self.block_size = block_size

        self.get_chunk_max = lambda chunk: torch.max(
            torch.abs(chunk), dim=-1, keepdim=True
        )[0]

    def cast(self, x: torch.Tensor, block_dim: int) -> torch.Tensor:
        _x = x.float().transpose(block_dim, -1)  # dim swap
        xshape = _x.shape  # remember shape
        _xs = torch.split(
            _x.reshape((-1, xshape[-1])), self.block_size, dim=-1
        )  # slice to chunks
        _ss = [
            2 ** torch.floor(torch.log2(self.get_chunk_max(chunk)))
            / self.element_format.largest_representable_power_of_two
            for chunk in _xs
        ]
        _x = torch.cat(
            [
                self.element_format.cast(chunk / scale) * scale
                for chunk, scale in zip(_xs, _ss)
            ],
            dim=block_dim,
        )  # quantize
        _x = _x.reshape(xshape).transpose_(block_dim, -1)  # recover shape
        return _x

    @property
    def bytes_per_elem(self) -> float:
        return (
            self.element_format.bytes_per_elem
            + self.scaler_format.bytes_per_elem / self.block_size
        )

    @property
    def bit_precision(self) -> float:
        return self.element_format.precision + 8.0 / self.block_size

    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse(
            "MXFP{precision:d}[E{exponent:d}M{mantissa:d}]{{{block_size:d}}}",
            sh,
        )
        assert conf["precision"] == conf["exponent"] + conf["mantissa"] + 1
        return cls(
            element_format=FloatingPoint(
                mantissa=conf["mantissa"],
                exponent=conf["exponent"],
                bias=2 ** (conf["exponent"] - 1) - 1,
                flush_subnormal=False,
                unsigned=False,
                rounding="nearest",
            ),
            block_size=conf["block_size"],
        )

    def __str__(self) -> str:
        return f"Simulated MXFP format: element format = {self.element_format}, scaler format = {self.scaler_format},\n block size = {self.block_size}"

    def __repr__(self) -> str:
        return f"MXFP{self.element_format.exponent+self.element_format.mantissa+1}[E{self.element_format.exponent}M{self.element_format.mantissa}]{{{self.block_size}}}"

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.element_format,
                self.block_size,
            ),
        )


class MXINT(BlockFloatingPoint):
    r"""
    This is a MXINT tensor format.
    """

    def __init__(
        self,
        precision=8,
        block_size=32,
    ):
        super().__init__(
            precision=precision,
            block_size=block_size,
            symmetric=True,
            rounding="nearest",
        )

    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse(
            "MXINT{precision:d}{{{block_size:d}}}",
            sh,
        )
        return cls(
            precision=conf["precision"],
            block_size=conf["block_size"],
        )

    def __str__(self) -> str:
        return f"Simulated MXINT format: precision bits = {self.precision}, block size = {self.block_size}\ncasting behavior: symmetric = {self.symmetric}, rounding = {self.rounding}"

    def __repr__(self) -> str:
        return f"MXINT{self.precision}{{{self.block_size}}}"

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.precision,
                self.block_size,
            ),
        )
