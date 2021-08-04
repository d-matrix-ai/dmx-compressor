from parse import parse
import torch
import torch.nn as nn
import torch.nn.functional as F


__ALL__ = [
    "ApproximationFunction",
    "ApproximationMixin",
    "NoApproximation",
    "SoftmaxApproximation",
    "GELUApproximation",
    "LayerNormApproximation",
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
        assert algorithm in ("poly2", "base2"), f"unsupported softmax algorithm {algorithm}"
        assert nform in ("int", "float32", "float16", "bfloat16"), f"unsupported softmax intermediate numerical format {nform}"

        self.algorithm = algorithm
        self.nform = nform

    def execute(self, *args, **kwargs):
        return eval(f"{self.algorithm}softmax")(
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

    def __init__(self):
        super().__init__()

    def execute(self, *args, **kwargs):
        #TODO: implement this
        return None

    @classmethod
    def from_shorthand(cls, sh: str):
        #TODO: implement this
        return cls()

    def __str__(self) -> str:
        return f"Layernorm approximation function"

    def __repr__(self) -> str:
        return f"LAYERNORM"


class GELUApproximation(ApproximationFunction):
    r"""
    This class specifies an approximation function for gelu nonlinearity.
    """

    def __init__(self):
        super().__init__()

    def execute(self, *args, **kwargs):
        #TODO: implement this
        return None

    @classmethod
    def from_shorthand(cls, sh: str):
        #TODO: implement this
        return cls()

    def __str__(self) -> str:
        return f"GELU approximation function"

    def __repr__(self) -> str:
        return f"GELU"


class Approximator(nn.Module):
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
        self.approximator = Approximator()
        self.approximation_error = None

    def _forward(self, input, *args, **kwargs):
        _output = super().forward(input)
        if not isinstance(self.approximator.function, NoApproximation):
            with torch.no_grad():
                _approx = self.approximator(input, *args, **kwargs)
                self.approximation_error = _approx - _output.data
                _output.data = _approx
        return _output


def poly2softmax(x, dim=-1, nform="float16", **kwargs):
    r"""
    This function computes Softmax using a range
    reduction technique and second order polynomial approximation
    for exp(r), where r is in reduced range. Various numerical
    formats can be specified using nform variable,
    including int (fixed point), float32, float16, and bfloat16
    """
    eps = 1.0e-30  # small number to avoid dividing by zero

    # compute exp(r) and k, where exp(x)=2^k * exp(r)
    ey, k = poly2exp(x, nform=nform, dim=dim)
    kmax, _ = torch.max(k, dim=dim, keepdim=True)

    # compute sum with normalization for numerical stability
    ey = ey * 2 ** (k - kmax)
    sum_exp = torch.sum(ey, dim=dim, keepdim=True)

    # compute softmax
    y = ey / (sum_exp + eps)

    return y


def poly2exp(x, nform, dim=-1):
    r"""
    This function computes exp(x) using a range reduction
    technique x = r + k*log(2), where k is an integer and
    -log(2)/2 < r < +log(2)/2.  exp(x) = 2^k * exp(r),
    where exp(r) is approximated by a second degree polynomial
    """
    ln2 = 0.69315  # log(2)
    invln2 = 1.4427  # 1 / log(2)
    # polynomial coefficients
    c0f = 1.0
    c1f = 1.015082
    c2f = 0.503765

    if nform == "int":
        scale = 14  # emulate integer arithmetic with 14 bit fractional part
        c0int = 2 ** scale  # poly coefficient c0 = 1
        c1int = round(1.0151 * 2 ** scale)  # poly coefficient c1
        # note poly coefficient c2 is 0.5

        # range reduction to range -log(2)/2 < r < log(2)/2
        k = torch.round(x * invln2)
        r = x - k * ln2

        # compute exp(r) emulating fixed point arithmetic
        rint = torch.round(r * 2 ** scale)

        mult_add1 = torch.round(c1int + 0.5 * rint)
        mult_add2 = torch.round(c0int + torch.round(mult_add1 * rint * 2 ** (-scale)))
        ey = mult_add2 * 2 ** (-scale)  # convert back to decimal number

    elif nform == "float32":
        ln2 = torch.tensor(ln2, dtype=torch.float32)
        invln2 = torch.tensor(invln2, dtype=torch.float32)
        c0 = torch.tensor(c0f, dtype=torch.float32)
        c1 = torch.tensor(c1f, dtype=torch.float32)
        c2 = torch.tensor(c2f, dtype=torch.float32)
        xfp32 = x.float()
        # range reduction to range -log(2)/2 < r < log(2)/2
        k = torch.round(xfp32 * invln2)
        r = xfp32 - k * ln2

        # compute exp(r) in FL32
        mult_add1 = c1 + c2 * r
        ey = c0 + mult_add1 * r

    elif nform == "float16":
        ln2 = torch.tensor(ln2, dtype=torch.float16)
        invln2 = torch.tensor(invln2, dtype=torch.float16)
        c0 = torch.tensor(c0f, dtype=torch.float16)
        c1 = torch.tensor(c1f, dtype=torch.float16)
        c2 = torch.tensor(c2f, dtype=torch.float16)

        xfp16 = x.half()
        # range reduction to range -log(2)/2 < r < log(2)/2
        xtmp = xfp16 * invln2
        k = torch.round(xtmp.float())
        k = k.half()
        r = xfp16 - k * ln2

        # compute exp(r) in FL16
        mult_add1 = c1 + c2 * r
        ey = c0 + mult_add1 * r

    elif nform == "bfloat16":
        ln2 = torch.tensor(ln2, dtype=torch.bfloat16)
        invln2 = torch.tensor(invln2, dtype=torch.bfloat16)
        c0 = torch.tensor(c0f, dtype=torch.bfloat16)
        c1 = torch.tensor(c1f, dtype=torch.bfloat16)
        c2 = torch.tensor(c2f, dtype=torch.bfloat16)

        xfp16 = x.bfloat16()
        # range reduction to range -log(2)/2 < r < log(2)/2
        xtmp = xfp16 * invln2
        k = torch.round(xtmp.float())
        k = k.bfloat16()
        r = xfp16 - k * ln2

        # compute exp(r) in BFL32
        mult_add1 = c1 + c2 * r
        ey = c0 + mult_add1 * r

    else:
        raise RuntimeError("unsuported numerical format")

    ey = ey.float()
    k = k.float()
    return ey, k


def base2softmax(x, dim=-1, nform="float16", **kwargs):
    r"""
    This function computes Softmax using base2exp
    function for the exp(x). Various numerical
    formats can be specified using nform variable,
    including int (fixed point), float32, float16, and bfloat16
    """
    eps = 1.0e-30  # small number to avoid dividing by zero
    # compute exp(x) for input vector x
    # including integer vector k for normalization
    ey, k = base2exp(x, nform=nform, dim=dim)

    kmax, _ = torch.max(k, dim=dim, keepdim=True)

    # compute sum with normalization for numerical stability
    ey = ey * 2 ** (k - kmax)
    sum_ey = torch.sum(ey, dim=dim, keepdim=True)

    # compute softmax
    y = ey / (sum_ey + eps)
    return y


def base2exp(x, nform, dim=-1):
    r"""
    This function computes exp(x) using a range reduction
    technique exp(x)=(2^k)*2^v, where k is an integer and 0<v<1
    2^v is approximated by a simple linear interpolation d+v.
    nform specifies the numerical format.
    """

    log2e_fp = 1.4426950408889634
    # log2(e)
    d = 0.957  # minmax solution for d over the input range 0<v<1

    if nform == "int":
        scale = 14  # assuming 14 bits after binary fixed point
        log2e = round(log2e_fp * 2 ** scale) / 2 ** scale  # log2(e)
        d = round(d * 2 ** scale) / 2 ** scale

        # range reduction to k and v
        z = x * log2e
        k = torch.floor(z)
        v = z - k
        v = torch.round(v * 2 ** scale) / 2 ** scale
    elif nform == "float32":
        log2e = torch.tensor(log2e_fp, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)

        # range reduction to k and v
        xfp32 = x.float()
        z = xfp32 * log2e
        k = torch.floor(z)
        v = z - k
    elif nform == "float16":
        log2e = torch.tensor(log2e_fp, dtype=torch.float16)
        d = torch.tensor(d, dtype=torch.float16)
        # range reduction to k and v
        xfp16 = x.half()
        z = xfp16 * log2e
        k = torch.floor(z.float())  # floor not implemented for float16
        k = k.half()
        v = z - k
    elif nform == "bfloat16":
        log2e = torch.tensor(log2e_fp, dtype=torch.bfloat16)
        d = torch.tensor(d, dtype=torch.bfloat16)

        # range reduction to k and v
        xfp16 = x.bfloat16()
        z = xfp16 * log2e
        k = torch.floor(z.float())  # floor not implemented for float16
        k = k.bfloat16()
        v = z - k
    else:
        raise RuntimeError("unsuported numerical format")

    # compute exp(v)
    two_pow_v = v + d

    ey = two_pow_v.float()
    k = k.float()
    return ey, k
