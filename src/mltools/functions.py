import functools
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
        eps = torch.tensor(2**-24, dtype=torch.float16)
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
        c0int = 2**scale  # poly coefficient c0 = 1
        c1int = round(1.0151 * 2**scale)  # poly coefficient c1
        # note poly coefficient c2 is 0.5

        # range reduction to range -log(2)/2 < r < log(2)/2
        k = torch.round(x * invln2)
        r = x - k * ln2

        # compute exp(r) emulating fixed point arithmetic
        rint = torch.round(r * 2**scale)

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


def base2softmax(x, dim=-1, nform="float16", quake3=False, **kwargs):
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
    if quake3:
        assert (
            nform == "float16"
        ), "only float16 format has quake3 reciprocal implementation as of now"
    if nform == "float16":
        # compute sum in fl16
        ey = ey.half()
        sum_ey = torch.sum(ey, dim=dim, keepdim=True)

        # compute softmax
        eps = torch.tensor(2**-24, dtype=torch.float16)
        y = ey * recip_float16_quake3(sum_ey + eps) if quake3 else ey / (sum_ey + eps)
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


base2quake3softmax = functools.partial(base2softmax, quake3=True)


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
        log2e = round(log2e_fp * 2**scale) / 2**scale  # log2(e)
        d = round(d * 2**scale) / 2**scale

        # range reduction to k and v
        z = x * log2e
        k = torch.floor(z)
        v = z - k
        v = torch.round(v * 2**scale) / 2**scale
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


def recip_float16_quake3(xin):
    r"""
    This function computes an approximate reciprocal in FP16
    using the Quake III Algorithm
    """
    assert xin.dtype == torch.float16, "input must be a float16 tensor"
    # initial guess
    x0_int = 0x77A8 - xin.cpu().numpy().view(dtype=np.uint16)
    x0 = torch.from_numpy(x0_int.view(dtype=np.float16)).to(xin.device)
    # one iteration of Newton-Ralphson
    r1 = 1.0 - xin * x0
    x1 = x0 + r1 * x0

    return x1


def recip_sqrt_float16_quake3(xin):
    r"""
    This function computes an approximate sqrt reciprocal in FP16
    using the Quake III Algorithm
    """
    assert xin.dtype == torch.float16, "input must be a float16 tensor"
    # initial guess
    x0_int = 0x59BE - (xin.cpu().numpy().view(dtype=np.uint16) >> 1)
    x0 = torch.from_numpy(x0_int.view(dtype=np.float16)).to(xin.device)
    # one iteration of Newton-Ralphson
    xin2 = 0.5 * xin
    xin2 *= x0
    r1 = 1.5 - xin2 * x0
    x1 = r1 * x0

    return x1


def recip_sqrt_float32_quake3(xin):
    r"""
    This function computes an approximate sqrt reciprocal in FP32
    using the Quake III Algorithm
    """
    assert xin.dtype == torch.float32, "input must be a float32 tensor"
    # initial guess
    x0_int = 0x5F3759DF - (xin.cpu().numpy().view(dtype=np.uint32) >> 1)
    x0 = torch.from_numpy(x0_int.view(dtype=np.float32)).to(xin.device)
    # one iteration of Newton-Ralphson
    xin2 = 0.5 * xin
    xin2 *= x0
    r1 = 1.5 - xin2 * x0
    x1 = r1 * x0

    return x1


def quake3layer_norm(
    input, normalized_shape, weight=None, bias=None, eps=1e-5, nform="float16"
):
    r"""
    This is a custom implementation of torch.nn.functional.layer_norm() using the Quake III algorithm for reciprocal of squareroot computation.
    """
    if nform == "float16":
        _x = input.half()
        _xmean = torch.mean(
            _x, dim=tuple(range(-len(normalized_shape), 0)), keepdim=True
        )
        _xvar = torch.var(
            _x,
            dim=tuple(range(-len(normalized_shape), 0)),
            unbiased=False,
            keepdim=True,
        )
        _x = _x - _xmean
        if weight is not None:
            _x = _x * weight.half()
        _x = _x * recip_sqrt_float16_quake3(_xvar + max(np.finfo(np.float16).eps, eps))
        if bias is not None:
            _x = _x + bias.half()
    elif nform == "float32":
        _x = input.float()
        _xmean = torch.mean(
            _x, dim=tuple(range(-len(normalized_shape), 0)), keepdim=True
        )
        _xvar = torch.var(
            _x,
            dim=tuple(range(-len(normalized_shape), 0)),
            unbiased=False,
            keepdim=True,
        )
        _x = _x - _xmean
        if weight is not None:
            _x = _x * weight.float()
        _x = _x * recip_sqrt_float32_quake3(_xvar + max(np.finfo(np.float32).eps, eps))
        if bias is not None:
            _x = _x + bias.float()
    else:
        raise RuntimeError("unsuported numerical format")

    return _x.float()


def fallbacklayer_norm(
    input, normalized_shape, weight=None, bias=None, eps=2.0**-126, nform="float16"
):
    r"""
    This function computes layer norm in nform but with only 1/sqrt computed in FP32, a custom implementation of torch.nn.functional.layer_norm().
    """
    nform = eval(f"torch.{nform}")

    eps = torch.Tensor([eps]).to(torch.float32)
    # default eps==2.**-126 is the smallest normal FP32 number

    # compute mean and variance
    _x = input.to(nform)
    _xmean = torch.mean(_x, dim=tuple(range(-len(normalized_shape), 0)), keepdim=True)
    _xvar = torch.var(
        _x,
        dim=tuple(range(-len(normalized_shape), 0)),
        unbiased=False,
        keepdim=True,
    )

    # compute reciprocal sqrt of var in FP32
    _xvar_FP32 = (
        _xvar.to(torch.float32) + eps
    )  # cast FP32; + eps to avoid dividing by zero
    _xvar_sqrt_recip_FP32 = torch.ones(1, dtype=nform) / torch.sqrt(
        _xvar_FP32
    )  # compute sqrt reciprocal in FP32
    _xvar_sqrt_recip = _xvar_sqrt_recip_FP32.to(nform)  # convert back to nform

    # compute normalized output
    _x = _x - _xmean
    if weight is not None:
        _x = _x * weight.to(nform)
    _x = _x * _xvar_sqrt_recip
    if bias is not None:
        _x += bias.to(nform)

    return _x


def poly2gelu(xin, nform="float16"):
    r"""
    This function computes a 2nd-order polynomial approximaiton to gelu()
    """
    a1, a2, b1, b2, recip_sqrt2 = -0.1444, 0.1444, -1.769, 1.769, 0.70711
    if nform == "float16":
        xin = xin.half()
        x = xin * recip_sqrt2
        x_clip = torch.minimum(x.abs(), torch.Tensor([b2]).half().to(xin.device))
        r = x_clip + b1
        r = r * r
        L = torch.where(x >= 0, a1 * r + 1.0, a2 * r)
        y = xin * L
    else:
        raise RuntimeError("unsuported numerical format")

    return y.float()


def svd_lowrank_approximate_tensor(x, rank=6):
    r"""
    This function computes a low-rank approximaiton of an input tensor on the last 2 dimensions
    """
    m, n = x.shape[-2:]
    rank = min(rank, m, n)

    _x = x.reshape(-1, m, n)
    for i in range(len(_x)):
        u, s, vh = torch.linalg.svd(_x[i], full_matrices=False)
        s[rank:] = 0.0
        _x[i] = u @ torch.diag(s) @ vh
    return x

    # u, s, v = torch.svd_lowrank(x, q=rank)
    # return u @ s.diag_embed() @ v.transpose(-1, -2)


def svd_lowrank_linear(input, weight, bias=None, rank=6):
    r"""
    This function is an approximated version of torch.nn.functional.linear()
    """
    weight = svd_lowrank_approximate_tensor(weight, rank=rank)
    return torch.nn.functional.linear(input, weight, bias)


def svd_lowrank_conv2d(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, rank=6
):
    r"""
    This function is an approximated version of torch.nn.functional.conv2d()
    """
    weight = svd_lowrank_approximate_tensor(weight, rank=rank)
    return torch.nn.functional.conv2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
