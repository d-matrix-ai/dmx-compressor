import torch
import numpy as np


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

    # normalization for numerical stability
    ey = ey * 2 ** (k - kmax)

    # compute sum and softmax

    if nform == "float16":
        # compute sum in fl16
        ey = ey.half()
        sum_ey = torch.sum(ey, dim=dim, keepdim=True)

        # compute softmax
        eps = torch.tensor(2 ** -24, dtype=torch.float16)
        y = ey / (sum_ey + eps)
        y = y.float()

    elif nform == "bfloat16":
        # compute sum in bfl16
        ey = ey.bfloat16()
        sum_ey = torch.sum(ey, dim=dim, keepdim=True)

        # compute softmax
        eps = torch.tensor(eps, dtype=torch.bfloat16)
        y = ey / (sum_ey + eps)
        y = y.float()

    else:
        # compute sum in fl32
        sum_ey = torch.sum(ey, dim=dim, keepdim=True)

        # compute softmax
        y = ey / (sum_ey + eps)

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

    # normalization for numerical stability
    ey = ey * 2 ** (k - kmax)

    # compute sum and softmax

    if nform == "float16":
        # compute sum in fl16
        ey = ey.half()
        sum_ey = torch.sum(ey, dim=dim, keepdim=True)

        # compute softmax
        eps = torch.tensor(2 ** -24, dtype=torch.float16)
        y = ey / (sum_ey + eps)
        y = y.float()

    elif nform == "bfloat16":
        # compute sum in bfl16
        ey = ey.bfloat16()
        sum_ey = torch.sum(ey, dim=dim, keepdim=True)

        # compute softmax
        eps = torch.tensor(eps, dtype=torch.bfloat16)
        y = ey / (sum_ey + eps)
        y = y.float()

    else:
        # compute sum in fl32
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


def recip_sqrt_float16(xin):
    r"""
    This function computes an approximate sqrt reciprocal in FP16
    using the Quake III Algorithm
    """
    assert xin.dtype == torch.float16, "input must be a float16 tensor"
    # initial guess
    x0_int = 22974 - (xin.numpy().view(dtype=np.uint16) >> 1)
    x0 = torch.from_numpy(x0_int.view(dtype=np.float16))
    # one iteration of Newton-Ralphson
    xin2 = 0.5 * xin
    xin2 *= x0
    r1 = 1.5 - xin2 * x0
    x1 = r1 * x0

    return x1
