import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


__ALL__ = [
    "softmax"
]

def poly2softmax(x, dim=-1):
    with torch.no_grad():
        n = x.shape[dim]

        ln2 = 0.69315  # log(2)
        invln2 = 1.4427  # 1 / log(2)
        scale = 14  # for exp() poly approximation assume 16bit integer arithmetic 
                    # with 14 bit fractional part
        c0int = 2**scale  # poly coefficient c0 = 1
        c1int = round(1.0151 * 2**scale)  # poly coefficient c1
        # note poly coefficient c2 is 0.5 

        # y = torch.zeros_like(x) # initialize output vector

        # range reduction to range -log(2)/2 < r < log(2)/2
        k = torch.round(x * invln2)
        r = x - k * ln2

        # compute exp(r) emulating fixed point arithmetic 
        rint = torch.round(r * 2**scale)
        r2int = torch.round(rint * rint * 2**(-scale))
        mult_add1 = torch.round(c0int + torch.round(c1int * rint * 2**(-scale)))
        mult_add2 = torch.round(mult_add1 + 0.5 * r2int)
        y = mult_add2 * 2**(-scale)
        # shift by k bits for final result
        y *= 2**k

        # compute softmax (note will need to use 32 bits for sum in HW)
        sum_exp = torch.sum(y, dim=dim, keepdim=True)
        y /= sum_exp
        # y = torch.round(y * 256) / 256
        
    return y


def softmax(x, dim=-1):
    _x = F.softmax(x, dim=dim)
    # print(_x)
    _x.data = poly2softmax(x, dim=dim)
    # print(_x)
    return _x


# class Softmax(Function):
#     r"""
#     d-MATRiX custom softmax in pytorch
#     """

#     @staticmethod
#     def forward(ctx, input, dim):
#         return poly2softmax(input, dim=dim), F.softmax(input, dim=dim)

#     @staticmethod
#     def backward(ctx, _g, g_output):
#         return g_output, None
