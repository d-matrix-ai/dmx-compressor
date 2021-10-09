import pytest
import torch
import torch.nn.functional as F
from mltools import numerical


RANDOM_SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)


def test_castto_bfp16_1():
    n = 1000
    x = torch.randn((1, n), dtype=torch.float32)
    x *= 0.49 / x.abs().max()
    x += 1.0

    _x = numerical.CastTo(format="BFP[8|8]{1,-1}(N)")(x)
    assert torch.allclose(_x, x, rtol=0., atol=2**-7)

    x = -x

    _x = numerical.CastTo(format="BFP[8|8]{1,-1}(N)")(x)
    assert torch.allclose(_x, x, rtol=0., atol=2**-7)

