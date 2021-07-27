import torch
import torch.nn.functional as F


__ALL__ = [
    "ApproximationMixin",
]


class ApproximationMixin:
    r"""
    Mixin for modules with approximated forward logic
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.approximation_function = None
        self.approximation_error = None

    def _forward(self, input, *args, **kwargs):
        _output = super().forward(input)
        if self.approximation_function is not None:
            with torch.no_grad():
                _approx = eval(self.approximation_function)(input, *args, **kwargs)
                self.approximation_error = _approx - _output.data
                _output.data = _approx
        return _output

    def extra_repr(self):
        return (
            None
            if self.approximation_function is None
            else f"approximation_function = {self.approximation_function}"
        )


def poly2softmax(x, dim=-1, **kwargs):
    r"""
    An approximate integer algorithm to replace torch.nn.functional.softmax()
    """
    n = x.shape[dim]

    ln2 = 0.69315  # log(2)
    invln2 = 1.4427  # 1 / log(2)
    scale = 14  # for exp() poly approximation assume 16bit integer arithmetic
    # with 14 bit fractional part
    c0int = 2 ** scale  # poly coefficient c0 = 1
    c1int = round(1.0151 * 2 ** scale)  # poly coefficient c1
    # note poly coefficient c2 is 0.5

    # y = torch.zeros_like(x) # initialize output vector

    # range reduction to range -log(2)/2 < r < log(2)/2
    k = torch.round(x * invln2)
    r = x - k * ln2

    # compute exp(r) emulating fixed point arithmetic
    rint = torch.round(r * 2 ** scale)
    r2int = torch.round(rint * rint * 2 ** (-scale))
    mult_add1 = torch.round(c0int + torch.round(c1int * rint * 2 ** (-scale)))
    mult_add2 = torch.round(mult_add1 + 0.5 * r2int)
    y = mult_add2 * 2 ** (-scale)
    # shift by k bits for final result
    y *= 2 ** k

    # compute softmax (note will need to use 32 bits for sum in HW)
    sum_exp = torch.sum(y, dim=dim, keepdim=True)
    y /= sum_exp
    # y = torch.round(y * 256) / 256

    return y
