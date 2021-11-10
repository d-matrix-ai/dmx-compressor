from parse import parse
import torch
import torch.nn as nn
import torch.nn.functional as F
from mltools import functions

__ALL__ = [
    "ApproximationFunction",
    "ApproximationMixin",
    "NoApproximation",
    "SoftmaxApproximation",
    "GELUApproximation",
    "LayerNormApproximation",
    "Approximate",
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
        return f"Dummy approximation function: no approximation"

    def __repr__(self) -> str:
        return f"NONE"


class SoftmaxApproximation(ApproximationFunction):
    r"""
    This class specifies an approximation function for softmax.
    """

    def __init__(self, algorithm="base2", nform="float16"):
        super().__init__()
        # check validity of configuration
        assert algorithm in (
            "poly2",
            "base2",
            "base2quake3",
        ), f"unsupported softmax algorithm {algorithm}"
        assert nform in (
            "int",
            "float32",
            "float16",
            "bfloat16",
        ), f"unsupported softmax intermediate numerical format {nform}"

        self.algorithm = algorithm
        self.nform = nform

    def execute(self, *args, **kwargs):
        return eval(f"functions.{self.algorithm}softmax")(
            *args, **dict(kwargs, nform=self.nform)
        )

    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse("SOFTMAX({algorithm:w},{nform:w})", sh)
        return cls(
            algorithm=conf["algorithm"],
            nform=conf["nform"],
        )

    def __str__(self) -> str:
        return f"Softmax approximation function: algorithm = {self.algorithm}, nform = {self.nform}"

    def __repr__(self) -> str:
        return f"SOFTMAX({self.algorithm},{self.nform})"


class LayerNormApproximation(ApproximationFunction):
    r"""
    This class specifies an approximation function for layer normalization.
    """

    def __init__(self, algorithm="quake3", nform="float16"):
        super().__init__()
        # check validity of configuration
        assert algorithm in ("quake3",), f"unsupported layer_norm algorithm {algorithm}"
        assert nform in (
            "float16",
            "float32",
        ), f"unsupported layer_norm numerical format {nform}"

        self.algorithm = algorithm
        self.nform = nform

    def execute(self, *args, **kwargs):
        return eval(f"functions.{self.algorithm}layer_norm")(
            *args, **dict(kwargs, nform=self.nform)
        )

    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse("LAYERNORM({algorithm:w},{nform:w})", sh)
        return cls(
            algorithm=conf["algorithm"],
            nform=conf["nform"],
        )

    def __str__(self) -> str:
        return f"Layernorm approximation function: algorithm = {self.algorithm}, nform = {self.nform}"

    def __repr__(self) -> str:
        return f"LAYERNORM({self.algorithm},{self.nform})"


class GELUApproximation(ApproximationFunction):
    r"""
    This class specifies an approximation function for gelu nonlinearity.
    """

    def __init__(self, algorithm="poly2", nform="float16"):
        super().__init__()
        # check validity of configuration
        assert algorithm in ("poly2",), f"unsupported layer_norm algorithm {algorithm}"
        assert nform in ("float16",), f"unsupported layer_norm numerical format {nform}"

        self.algorithm = algorithm
        self.nform = nform

    def execute(self, *args, **kwargs):
        return eval(f"functions.{self.algorithm}gelu")(
            *args,
            **dict(kwargs, nform=self.nform),
        )

    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse("GELU({algorithm:w},{nform:w})", sh)
        return cls(
            algorithm=conf["algorithm"],
            nform=conf["nform"],
        )

    def __str__(self) -> str:
        return f"GELU approximation function: algorithm = {self.algorithm}, nform = {self.nform}"

    def __repr__(self) -> str:
        return f"GELU({self.algorithm},{self.nform})"


class Approximate(nn.Module):
    r"""
    An approximation operator container
    """

    def __init__(self, function=NoApproximation()):
        super().__init__()
        if not isinstance(function, ApproximationFunction):
            function = ApproximationFunction.from_shorthand(function)
        self.function = function

    def forward(self, input, *args, **kwargs):
        return self.function.execute(input, *args, **kwargs)

    def extra_repr(self):
        return f"function = {self.function.__repr__()}"


class ApproximationMixin:
    r"""
    Mixin for modules with approximated forward logic
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.approximator = (
            Approximate()
        )  # if isinstance(self, CorsairModule) else None
        self.approximation_error = None

    def approx_forward(self, input, *args, **kwargs):
        _output = super().forward(input)
        if not isinstance(self.approximator.function, NoApproximation):
            with torch.no_grad():
                _approx = self.approximator(input, *args, **kwargs)
                self.approximation_error = _approx - _output.data
                _output.data = _approx
        return _output
