import torch
import qtorch
from qtorch.quant import fixed_point_quantize, float_quantize, block_quantize, quantizer
from qtorch import FixedPoint, FloatingPoint, BlockFloatingPoint
import numpy as np
from functools import partial


fp_exp = 8
fp_man = 23
fp_rnd = 'nearest'
fp = qtorch.FloatingPoint(exp=fp_exp, man=fp_man)

q = quantizer(forward_number=fp, forward_rounding=fp_rnd)

q_ = partial(float_quantize, exp=fp_exp, man=fp_man, rounding=fp_rnd)

x = torch.randn(8, 8)

print(x)
print(q(x))
print(q_(x))

import ipdb; ipdb.set_trace()
assert np.allclose(q(x).numpy(), q_(x).numpy(), rtol=1e-6)
assert np.allclose(x.numpy(), q_(x).numpy(), rtol=1e-6)



# bfp_wl = 8
# bfp_dim = 1
# bfp_rnd = 'nearest'
# bfp = qtorch.BlockFloatingPoint(wl=bfp_wl, dim=bfp_dim)

# q = quantizer(forward_number=bfp, forward_rounding=bfp_rnd)

# q_ = partial(block_quantize, wl=bfp_wl, dim=bfp_dim, rounding=bfp_rnd)

# x = torch.randn(8, 8)

# print(x)
# print(q(x))
# print(q_(x))

# assert np.allclose(q(x).numpy(), q_(x).numpy(), rtol=1e-6)
# assert np.allclose(x.numpy(), q_(x).numpy(), rtol=1e-3)
# import ipdb; ipdb.set_trace()

ma, ex = np.frexp(q(x))

# xp_wl = 24
# xp_fl = 6
# xp_rnd = 'stochastic'
# xp = qtorch.FixedPoint(wl=xp_wl, fl=xp_fl)

# q = quantizer(forward_number=xp, forward_rounding=xp_rnd)

# q_ = partial(fixed_point_quantize, wl=xp_wl, fl=xp_fl, rounding=xp_rnd)

# x = torch.randn(8, 8)

# print(x)
# print(q(x))
# print(q_(x))

