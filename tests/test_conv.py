import pytest
import torch
from compressor import dmx


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
    ),
)
@pytest.mark.parametrize(
    "batch_size",
    (
        1,
        16,
    ),
)
@pytest.mark.parametrize(
    "module_type",
    (
        "Conv2d",
        "ConvTranspose2d",
    ),
)
def test_conv2d(
    module_type,
    batch_size,
    image_size,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    bias,
):
    torch_module = eval(f"torch.nn.{module_type}")(
        in_channels, out_channels, kernel_size, stride=stride, bias=bias, device=device
    )
    dmx_module = eval(f"dmx.nn.{module_type}")(
        in_channels, out_channels, kernel_size, stride=stride, bias=bias, device=device
    )
    dmx_module.weight.data = torch_module.weight.data
    if bias:
        dmx_module.bias.data = torch_module.bias.data
    t_inp = torch.randn(
        batch_size, in_channels, *image_size, device=device
    ).requires_grad_()
    c_inp = t_inp.clone().detach().requires_grad_()
    t_out = torch_module(t_inp)
    c_out = dmx_module(c_inp)
    g_out = torch.randn_like(t_out)
    t_out.backward(g_out)
    c_out.backward(g_out)
    assert torch.allclose(t_out.data, c_out.data, atol=1e-6)
    assert torch.allclose(t_inp.grad, c_inp.grad, atol=1e-6)
