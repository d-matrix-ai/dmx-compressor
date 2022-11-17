import pytest
import torch
import torch.nn.functional as F
from mltools import numerical


RANDOM_SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)


def test_castto_bfp16_1():
    n = 1000
    x = torch.randn((1, n), dtype=torch.float32).to(device)
    x *= 0.5 / x.abs().max()
    x += 1.0

    _x = numerical.CastTo(format="BFP[8|8]{1,-1}(SN)")(x)
    assert torch.allclose(_x, x, rtol=0.0, atol=2 ** -7)

    x = -x

    _x = numerical.CastTo(format="BFP[8|8]{1,-1}(SN)")(x)
    assert torch.allclose(_x, x, rtol=0.0, atol=2 ** -7)


def test_bfp16_1_rounding():
    x = torch.Tensor(
        [
            1.0,
            1.0 + 2 ** -7,
            1.0 + 2 ** -6,
            1.0 + 2 ** -6 + 2 ** -7,
        ]
    ).to(device)
    y = torch.Tensor(
        [
            1.0,
            1.0,
            1.015625,
            1.03125,
        ]
    ).to(device)
    assert torch.all(numerical.CastTo(format="BFP[8|8]{1,-1}(SN)")(x) == y)
    assert torch.all(numerical.CastTo(format="BFP[8|8]{1,-1}(SN)")(-x) == -y)


def test_bfp12_1_rounding():
    x = torch.Tensor(
        [
            1.0,
            1.0 + 2 ** -3,
            1.0 + 2 ** -2,
            1.0 + 2 ** -2 + 2 ** -3,
        ]
    ).to(device)
    y = torch.Tensor(
        [
            1.0,
            1.0,
            1.25,
            1.5,
        ]
    ).to(device)
    assert torch.all(numerical.CastTo(format="BFP[4|8]{1,-1}(SN)")(x) == y)
    assert torch.all(numerical.CastTo(format="BFP[4|8]{1,-1}(SN)")(-x) == -y)
