from typing import Union, Optional, Callable
from parse import parse
from bidict import bidict
import torch
import torch.nn as nn
from .functions import *
import transformers

try:
    from dmx.common.vsimd.ported.operator import functional as vsimd

    VSIMD_OP_REF_AVAILABLE = True
except ModuleNotFoundError:
    VSIMD_OP_REF_AVAILABLE = False

torch_function_mapping = bidict(
    {
        "GELU": torch.nn.functional.gelu,
        "SILU": torch.nn.functional.silu,
        "RMS_NORM": torch.nn.functional.rms_norm,
        "LAYER_NORM": torch.nn.functional.layer_norm,
        "SOFTMAX": torch.nn.functional.softmax,
    }
)

custom_function_mapping = bidict(
    {
        "APPLY_LLAMA_ROPE": transformers.models.llama.modeling_llama.apply_rotary_pos_emb,
    }
)

__ALL__ = [
    "ApproximationFunction",
    "ApproximationMixin",
    "NoApproximation",
    "TorchFunctionApproximation",
    "Approximate",
    "Approximator",
]


class ApproximationFunction:
    r"""
    This is an abstract class of approximation algorithm.
    Child classes to implement `execute()` and `from_shorthand()` method.
    """

    def __init__(self):
        pass

    def __str__(self) -> str:
        raise NotImplementedError

    def execute(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def from_shorthand(sh: str):
        if sh.startswith("NONE"):
            return NoApproximation.from_shorthand(sh)
        elif sh.startswith((*torch_function_mapping.keys(),)):
            return TorchFunctionApproximation.from_shorthand(sh)
        elif sh.startswith((*custom_function_mapping.keys(),)):
            return CustomFunctionApproximation.from_shorthand(sh)
        else:
            raise ValueError(f"unrecognized approximation function shorthand: {sh}")


class NoApproximation(ApproximationFunction):
    r"""
    This is a dummy approximation algorithm that means no approximation.
    """

    def __init__(self):
        super().__init__()

    def execute(self, *args, **kwargs):
        raise RuntimeError("NoApproximation is not supposed to be executed")

    @classmethod
    def from_shorthand(cls, sh: str):
        return cls()

    def __str__(self) -> str:
        return "Dummy approximation function: no approximation"

    def __repr__(self) -> str:
        return "NONE"


Identity = NoApproximation  # an alias, to be deprecated


class TorchFunctionApproximation(ApproximationFunction):
    r"""
    This class specifies an approximation function for a member of torch.nn.functional.
    """

    def __init__(self, func_id: None, algorithm: str = "vsimd", **extra_params):
        super().__init__()
        self.func_id = func_id
        self.torch_functional = torch_function_mapping[func_id]
        self.algorithm = algorithm
        self.extra_params = extra_params

    @classmethod
    def from_shorthand(cls, sh: str):
        from dmx.compressor.utils.io import string_to_kwargs

        conf = parse("{func_ID:w}[{algorithm:w}]({extra_params:w})", sh)
        _op = torch_function_mapping(conf["func_ID"])
        _algo = conf["algorithm"]
        _extra_params = string_to_kwargs(conf["extra_params"])
        return cls(torch_functional=_op, algorithm=_algo, **_extra_params)

    def execute(self, *args, **kwargs):
        if self.algorithm == "vsimd":
            assert VSIMD_OP_REF_AVAILABLE, "SIMD op reference not available"
            return vsimd.gelu(*args, **kwargs, **self.extra_params)
        elif self.algorithm in ["experimental"]:
            return eval(self.algorithm).gelu(*args, **kwargs, **self.extra_params)
        else:
            raise ValueError(f"unknown approximation algorithm {self.algorithm} for {self.func_id}")

    def __str__(self) -> str:
        return f"Approximated version of {self.torch_functional} with annotation: {self.__repr__()}"

    def __repr__(self) -> str:
        from dmx.compressor.utils.io import kwargs_to_string

        return f"{self.func_id}[{self.algorithm}]({kwargs_to_string(**self.extra_params)})"


class CustomFunctionApproximation(ApproximationFunction):
    r"""
    This class specifies an approximation function for a custom written torch function.
    """
    pass


class Approximate(nn.Module):
    r"""
    An approximation operator container
    """

    def __init__(self, function=NoApproximation()):
        super().__init__()
        self.set_function(function)

    def set_function(self, function: Union[str, ApproximationFunction]) -> None:
        if not isinstance(function, ApproximationFunction):
            function = ApproximationFunction.from_shorthand(function)
        self.function = function

    def forward(self, input, *args, **kwargs):
        return self.function.execute(input, *args, **kwargs)

    def extra_repr(self):
        return f"function = {self.function.__repr__()}"


class Approximator(nn.Module):
    r"""
    Encapsulates approximation forward logic of a module
    """

    def __init__(self, function=Identity()):
        super().__init__()
        if not isinstance(function, ApproximationFunction):
            function = ApproximationFunction.from_shorthand(function)
        self.function = function

    def forward(self, input):
        _output = self.function.execute(input)[0]
        if not isinstance(
            self.function,
            (NoApproximation,),
        ):
            with torch.no_grad():
                self.approximation_error = _output - input
        return _output

    def extra_repr(self):
        return f"function = {self.function.__repr__()}"

    def approximation_function(self):
        return repr(self.function)


class ApproximationMixin:
    r"""
    Mixin to equip modules with approximated forward logic through Approximator
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.approximator = Approximate()  # if isinstance(self, DmxModule) else None
        self.approximation_error = None

    def approx_forward(self, inputs, *args, **kwargs):
        if not self.functional_forward is None:
            _output = self.functional_forward(*inputs, *args, **kwargs)
        else:
            _output = super().forward(*inputs)
        if not isinstance(
            self.approximator.function,
            (NoApproximation,),
        ):
            with torch.no_grad():
                _approx = self.approximator(*inputs, *args, **kwargs)
                self.approximation_error = _approx - _output.data
                _output.data = _approx
        return _output

    @property
    def approximation_function(self):
        return self.approximator.function
