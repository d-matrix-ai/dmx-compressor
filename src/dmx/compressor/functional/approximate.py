from typing import Union, Optional
from parse import parse
import torch
import torch.nn as nn
from . import vsimd


__ALL__ = [
    "ApproximationFunction",
    "ApproximationMixin",
    "NoApproximation",
    "SoftmaxApproximation",
    "GELUApproximation",
    "LayerNormApproximation",
    "HFDiffusersTimestepsApproximation",
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
        elif sh.startswith("SOFTMAX"):
            return SoftmaxApproximation.from_shorthand(sh)
        elif sh.startswith("GELU"):
            return GELUApproximation.from_shorthand(sh)
        elif sh.startswith("LAYERNORM"):
            return LayerNormApproximation.from_shorthand(sh)
        elif sh.startswith("HFDIFFUSERSTIMESTEPS"):
            return HFDiffusersTimestepsApproximation.from_shorthand(sh)
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


class Identity(ApproximationFunction):
    r"""
    This is a identity function that does no approximation and just returns the original value
    """

    def __init__(self, algorithm="identity"):
        super().__init__()
        self.algorithm = algorithm

    def execute(self, *args, **kwargs):
        return args

    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse("Identity({algorithm:w})", sh)
        return cls(
            algorithm=conf["algorithm"],
        )

    def __str__(self) -> str:
        return "Identity: return itself"

    def __repr__(self) -> str:
        return "Identity"


class SoftmaxApproximation(ApproximationFunction):
    r"""
    This class specifies an approximation function for softmax.
    """

    def __init__(self, algorithm: str = "vsimd", nform: Optional[str] = None):
        super().__init__()
        # check validity of configuration
        assert algorithm in (
            "vsimd",
            "poly2",
            "base2",
            "base2quake3",
        ), f"unsupported softmax algorithm {algorithm}"
        assert nform in (
            "int",
            "float32",
            "float16",
            "bfloat16",
            None,
        ), f"unsupported softmax intermediate numerical format {nform}"

        self.algorithm = algorithm
        self.nform = nform

    def execute(self, *args, **kwargs):
        if self.algorithm == "vsimd":
            return vsimd.softmax(*args, **kwargs)
        else:
            return eval(f"functions.{self.algorithm}softmax")(
                *args, **dict(kwargs, nform=self.nform)
            )

    @classmethod
    def from_shorthand(cls, sh: str):
        if sh == "SOFTMAX(vsimd)":
            return cls(algorithm="vsimd", nform=None)
        else:
            conf = parse("SOFTMAX({algorithm:w},{nform:w})", sh)
            return cls(
                algorithm=conf["algorithm"],
                nform=conf["nform"],
            )

    def __str__(self) -> str:
        return f"Softmax approximation function: algorithm = {self.algorithm}, nform = {self.nform}"

    def __repr__(self) -> str:
        return (
            "SOFTMAX(vsimd)"
            if self.algorithm == "vsimd"
            else f"SOFTMAX({self.algorithm},{self.nform})"
        )


class LayerNormApproximation(ApproximationFunction):
    r"""
    This class specifies an approximation function for layer normalization.
    """

    def __init__(
        self,
        algorithm: str = "vsimd",
        norm: Optional[int] = None,
        nform: Optional[str] = None,
    ):
        super().__init__()
        assert algorithm in (
            "vsimd",
            "fallback",
            "legacy",
            "quake3",
        ), f"unsupported layer_norm algorithm {algorithm}"
        assert nform in (
            "float16",
            "float32",
            None,
        ), f"unsupported layer_norm numerical format {nform}"

        self.algorithm = algorithm
        self.nform = nform
        self.norm = norm

    def execute(self, *args, **kwargs):
        if self.algorithm == "vsimd":
            return vsimd.layer_norm(*args, **kwargs)
        else:
            return eval(f"functions.{self.algorithm}layer_norm")(
                *args, **dict(kwargs, norm=self.norm, nform=self.nform)
            )

    @classmethod
    def from_shorthand(cls, sh: str):
        if sh == "LAYERNORM(vsimd)":
            return cls(algorithm="vsimd", norm=None, nform=None)
        else:
            conf = parse("LAYERNORM({algorithm:w},{norm},{nform:w})", sh)
            return cls(
                algorithm=conf["algorithm"],
                norm=conf["norm"],
                nform=conf["nform"],
            )

    def __str__(self) -> str:
        return f"Layernorm approximation function: algorithm = {self.algorithm}, norm = {self.norm}, nform = {self.nform}"

    def __repr__(self) -> str:
        return (
            "LAYERNORM(vsimd)"
            if self.algorithm == "vsimd"
            else f"LAYERNORM({self.algorithm},{self.norm},{self.nform})"
        )


class GELUApproximation(ApproximationFunction):
    r"""
    This class specifies an approximation function for gelu nonlinearity.
    """

    def __init__(self, algorithm: str = "vsimd", nform: Optional[str] = None):
        super().__init__()
        assert algorithm in (
            "vsimd",
            "poly2",
        ), f"unsupported gelu algorithm {algorithm}"
        assert nform in (
            "float16",
            None,
        ), f"unsupported gelu numerical format {nform}"

        self.algorithm = algorithm
        self.nform = nform

    def execute(self, *args, **kwargs):
        if self.algorithm == "vsimd":
            return vsimd.gelu(*args, **kwargs)
        else:
            return eval(f"functions.{self.algorithm}gelu")(
                *args,
                **dict(kwargs, nform=self.nform),
            )

    @classmethod
    def from_shorthand(cls, sh: str):
        if sh == "GELU(vsimd)":
            return cls(algorithm="vsimd", nform=None)
        else:
            conf = parse("GELU({algorithm:w},{nform:w})", sh)
            return cls(
                algorithm=conf["algorithm"],
                nform=conf["nform"],
            )

    def __str__(self) -> str:
        return f"GELU approximation function: algorithm = {self.algorithm}, nform = {self.nform}"

    def __repr__(self) -> str:
        return (
            "GELU(vsimd)"
            if self.algorithm == "vsimd"
            else f"GELU({self.algorithm},{self.nform})"
        )


class HFDiffusersTimestepsApproximation(ApproximationFunction):
    r"""
    This class specifies an approximation function for HFDiffusersTimesteps.
    """

    def __init__(self, algorithm: str = "vsimd", nform: Optional[str] = None):
        super().__init__()
        assert algorithm in (
            "vsimd",
        ), f"unsupported HF_DIFFUSERS_TIMESTEPS algorithm {algorithm}"
        assert nform in (
            "float16",
            None,
        ), f"unsupported HF_DIFFUSERS_TIMESTEPS numerical format {nform}"

        self.algorithm = algorithm
        self.nform = nform

    def execute(self, *args, **kwargs):
        if self.algorithm == "vsimd":
            return vsimd.hf_diffusers_timesteps(*args, **kwargs)
        else:
            raise ValueError("unrecognized the algorithm")

    @classmethod
    def from_shorthand(cls, sh: str):
        if sh == "HFDIFFUSERSTIMESTEPS(vsimd)":
            return cls(algorithm="vsimd", nform=None)
        else:
            conf = parse("HFDIFFUSERSTIMESTEPS({algorithm:w},{nform:w})", sh)
            return cls(
                algorithm=conf["algorithm"],
                nform=conf["nform"],
            )

    def __str__(self) -> str:
        return f"HFDIFFUSERSTIMESTEPS approximation function: algorithm = {self.algorithm}, nform = {self.nform}"

    def __repr__(self) -> str:
        return (
            "HFDIFFUSERSTIMESTEPS(vsimd)"
            if self.algorithm == "vsimd"
            else f"HFDIFFUSERSTIMESTEPS({self.algorithm},{self.nform})"
        )


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
    A nn.Module subclass that mimics the behavior of ApproximationMixin
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
    Mixin for modules with approximated forward logic
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.approximator = Approximate()  # if isinstance(self, DmxModule) else None
        self.approximation_error = None

    def approx_forward(self, inputs, *args, **kwargs):
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
