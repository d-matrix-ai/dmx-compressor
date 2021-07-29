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
    This function computes Softmax using a range
    reduction technique and second order polynomial approximation
    for exp(r), where r is in reduced range
    """
    eps = 1.0e-30  # small number to avoid dividing by zero

    #compute exp(r) and k, where exp(x)=2^k * exp(r)
    ey,k = poly2exp(x,dim=dim)

    kmax, _ = torch.max(k, dim=dim, keepdim=True)  # find max k along softmax dim.
    # shift by k-kmax bits for final result
    ey = ey * 2 ** (k - kmax)

    # compute softmax (note will need to use 32 bits for sum in HW)
    sum_exp = torch.sum(ey, dim=dim, keepdim=True)
    y = ey/(sum_exp + eps)

    return y

def poly2exp(x, dim=-1):
    r"""
    This function computes exp(x) using a range reduction
    technique x = r + k*log(2), where k is an integer and 
    -log(2)/2 < r < +log(2)/2.  exp(x) = 2^k * exp(r),
    where exp(r) is approximated by a second degree polynomial
    """
    ln2 = 0.69315  # log(2)
    invln2 = 1.4427  # 1 / log(2)
    scale = 14  # emulate bits after binary fixed point
    c0int = 2**scale  # poly coefficient c0 = 1
    c1int = round(1.0151 * 2**scale)  # poly coefficient c1
    # note poly coefficient c2 is 0.5 

    # range reduction to range -log(2)/2 < r < log(2)/2
    k = torch.round(x * invln2)
    r = x - k * ln2

    # compute exp(r) emulating fixed point arithmetic 
    rint = torch.round(r * 2**scale)
        
    mult_add1 = torch.round(c1int + 0.5*rint)
    mult_add2 = torch.round(c0int + torch.round(mult_add1*rint*2**(-scale)))
    ey = mult_add2 * 2**(-scale) #convert back to decimal number
    
    return ey, k

def base2softmax(x, dim=-1, **kwargs):
    r"""
    This function computes Softmax using base2exp
    function for the exp(x)
    """
    eps = 1.0e-30  # small number to avoid dividing by zero
    # compute exp(x) for input vector x
    # including integer vector k for re-normalization
    ey, k = base2exp(x, dim=dim)

    kmax, _ = torch.max(k, dim=dim, keepdim=True)  # find max k along softmax dim.

    # compute sum with normalization for numerical stability
    ey = ey * 2 ** (k - kmax)
    sum_ey = torch.sum(ey, dim=dim, keepdim=True)

    # compute softmax
    y = ey / (sum_ey + eps)

    return y


def base2exp(x, dim=-1):
    r"""
    This function computes exp(x) using a range reduction
    technique exp(x)=(2^k)*2^v, where k is an integer and 0<v<1
    2^v is approximated by a simple linear interpolation d+v.
    input x should be a vector shape (n,1)
    """

    scale = 14  # emulate bits after binary fixed point
    log2e_fp = 1.4426950408889634
    # log2(e) in floating point
    log2e = round(log2e_fp * 2 ** scale) / 2 ** scale  # log2(e)
    d = 0.957  # minmax solution for d over the input range 0<v<1
    d = round(d * 2 ** scale) / 2 ** scale

    # range reduction to k and v
    z = x * log2e
    k = torch.floor(z)
    v = z - k
    v = torch.round(v * 2 ** scale) / 2 ** scale

    # compute exp(x)
    two_pow_v = v + d

    ey = two_pow_v
    return ey, k
