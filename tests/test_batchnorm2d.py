import pytest
import torch
from mltools import dmx

RANDOM_SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)


@pytest.mark.parametrize("bsz", [1, 8])
@pytest.mark.parametrize("height", [64, 128])
@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("channels", [3, 64, 128])
@pytest.mark.parametrize("eps", [1e-5, 1e-12])
@pytest.mark.parametrize("affine", [True, False])
def test_batchnorm2d(bsz, height, width, channels, eps, affine):
    shape = (bsz, channels, height, width)
    x = torch.randn(*shape, device=device).requires_grad_()
    bn0 = torch.nn.BatchNorm2d(
        channels, eps=eps, affine=affine, momentum=0.1, track_running_stats=True
    ).to(device)
    bn1 = dmx.nn.BatchNorm2d(
        channels, eps=eps, affine=affine, momentum=0.1, track_running_stats=True
    ).to(device)

    if affine:
        gamma = torch.rand(channels, requires_grad=True, device=device)
        beta = torch.randn(channels, requires_grad=True, device=device)

        bn0.weight.data = gamma.data
        bn0.bias.data = beta.data

        bn1.weight.data = gamma.data
        bn1.bias.data = beta.data

    x0 = x.clone().detach().requires_grad_()
    x1 = x.clone().detach().requires_grad_()

    y0 = bn0(x0)
    y1 = bn1(x1)

    y1.backward(torch.ones_like(x, device=device))
    y0.backward(torch.ones_like(x, device=device))

    atol = 1e-12
    assert torch.allclose(y0, y1, atol=atol)
    assert torch.allclose(x0.grad, x1.grad, atol=atol)
