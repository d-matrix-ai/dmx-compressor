import pytest
import torch
import torch.nn.functional as F
from mltools import corsair


RANDOM_SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)


@pytest.mark.parametrize("bias", (True, False))
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride",
    (
        (1, 8, 5, 1),
        (3, 16, 5, 1),
        (3, 16, 3, 1),
        (64, 64, 3, 1),
        (64, 128, 3, 2),
    ),
)
@pytest.mark.parametrize(
    "image_size",
    (
        (28, 28),
        (32, 32),
        (224, 224),
        (256, 256),
    ),
)
@pytest.mark.parametrize(
    "batch_size",
    (
        1,
        16,
    ),
)
def test_conv2d(
    batch_size, image_size, in_channels, out_channels, kernel_size, stride, bias
):
    torch_module = torch.nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, bias=bias, device=device
    )
    corsair_module = corsair.nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, bias=bias, device=device
    )
    corsair_module.weight.data = torch_module.weight.data
    if bias:
        corsair_module.bias.data = torch_module.bias.data
    t_inp = (
        torch.randn(batch_size, in_channels, *image_size).to(device).requires_grad_()
    )
    c_inp = t_inp.clone().detach().requires_grad_()
    t_out = torch_module(t_inp)
    c_out = corsair_module(c_inp)
    g_out = torch.randn_like(t_out)
    t_out.backward(g_out)
    c_out.backward(g_out)
    assert torch.allclose(t_out.data, c_out.data, atol=1e-6)
    assert torch.allclose(t_inp.grad, c_inp.grad, atol=1e-6)
