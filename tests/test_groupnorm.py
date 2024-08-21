import pytest
import torch
from dmx.compressor.modeling import nn as dmxnn

RANDOM_SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)


@pytest.mark.parametrize("bsz", [1, 8])
@pytest.mark.parametrize("height", [64, 128])
@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("channels", [64, 128, 320])
@pytest.mark.parametrize("num_groups", [1, 2, 4, 8, 32])
@pytest.mark.parametrize("eps", [1e-5, 1e-12])
@pytest.mark.parametrize("affine", [True, False])
def test_groupnorm(bsz, height, width, channels, num_groups, eps, affine):
    shape = (bsz, channels, height, width)
    x = torch.randn(*shape, device=device).requires_grad_()
    gn0 = torch.nn.GroupNorm(num_groups, channels, affine=True, eps=eps).to(device)
    gn1 = dmxnn.GroupNorm(num_groups, channels, affine=True, eps=eps).to(device)

    if affine:
        gamma = torch.rand(channels, requires_grad=True, device=device)
        beta = torch.randn(channels, requires_grad=True, device=device)
        gn0.weight.data = gamma.data
        gn0.bias.data = beta.data
        gn1.weight.data = gamma.data
        gn1.bias.data = beta.data

    x0 = x.clone().detach().requires_grad_()
    x1 = x.clone().detach().requires_grad_()

    y0 = gn0(x0)
    y1 = gn1(x1)

    y1.backward(torch.ones_like(x, device=device))
    y0.backward(torch.ones_like(x, device=device))

    atol = 1e-12
    assert torch.allclose(y0, y1, atol=atol)
    assert torch.allclose(x0.grad, x1.grad, atol=atol)
