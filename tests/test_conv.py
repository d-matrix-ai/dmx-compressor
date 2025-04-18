import pytest
import torch
from dmx.compressor.modeling import nn as dmxnn
from dmx.compressor import format
from dmx.compressor.modeling.nn.experimental import Conv1dScatter, Conv1dUnfold, Conv2dGather, Conv2dUnfold
import dmx.ops


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
    dmx_module = eval(f"dmxnn.{module_type}")(
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
    "batch_size",
    (
        1,
        16,
    ),
)
@pytest.mark.parametrize("image_size_1d", (28, 32, 224))
def test_conv1d(
    batch_size, image_size_1d, in_channels, out_channels, kernel_size, stride, bias
):
    torch_module = eval(f"torch.nn.Conv1d")(
        in_channels, out_channels, kernel_size, stride=stride, bias=bias, device=device
    )
    dmx_module = eval(f"dmxnn.Conv1d")(
        in_channels, out_channels, kernel_size, stride=stride, bias=bias, device=device
    )
    dmx_module.weight.data = torch_module.weight.data
    if bias:
        dmx_module.bias.data = torch_module.bias.data
    t_inp = torch.randn(
        batch_size, in_channels, image_size_1d, device=device
    ).requires_grad_()
    c_inp = t_inp.clone().detach().requires_grad_()
    t_out = torch_module(t_inp)
    c_out = dmx_module(c_inp)
    g_out = torch.randn_like(t_out)
    t_out.backward(g_out)
    c_out.backward(g_out)
    assert torch.allclose(t_out.data, c_out.data, atol=1e-6)
    assert torch.allclose(t_inp.grad, c_inp.grad, atol=1e-6)


@pytest.mark.parametrize("bias", (True, False))
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride",
    (
        (80, 384, 3, 1),
        (384, 384, 3, 2),
    ),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize("image_size_1d", (30,))
def test_conv1d_scatter(
    batch_size, image_size_1d, in_channels, out_channels, kernel_size, stride, bias
):
    torch_module = eval(f"torch.nn.Conv1d")(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        bias=bias,
        device=device,
    )
    dmx_module = eval(f"dmxnn.Conv1d")(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        bias=bias,
        device=device,
    )
    dmx_module_scatter = eval(f"Conv1dScatter")(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        bias=bias,
        device=device,
    )
    dmx_module_unfold = eval(f"Conv1dUnfold")(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        bias=bias,
        device=device,
    )
    dmx_module.weight.data = torch_module.weight.data
    dmx_module_scatter.weight.data = torch_module.weight.data
    dmx_module_unfold.weight.data = torch_module.weight.data

    if bias:
        dmx_module.bias.data = torch_module.bias.data
        dmx_module_scatter.bias.data = torch_module.bias.data
        dmx_module_unfold.bias.data = torch_module.bias.data

    ### testing to_compiler_graph
    scatter_qdq = torch.fx.GraphModule(
        dmx_module_scatter, dmx_module_scatter.to_compiler_graph()
    )
    unfold_qdq = torch.fx.GraphModule(
        dmx_module_unfold, dmx_module_unfold.to_compiler_graph()
    )

    t_inp = torch.randn(
        batch_size, in_channels, image_size_1d, device=device
    ).requires_grad_()
    c_inp = t_inp.clone().detach().requires_grad_()
    c_inp_scatter = t_inp.clone().detach().requires_grad_()
    c_inp_unfold = t_inp.clone().detach().requires_grad_()
    t_out = torch_module(t_inp)
    c_out = dmx_module(c_inp)
    c_out_scatter = dmx_module_scatter(c_inp_scatter)
    c_out_unfold = dmx_module_unfold(c_inp_unfold)
    c_out_qdq_scatter = scatter_qdq(c_inp_scatter.clone().detach())
    c_out_qdq_unfold = unfold_qdq(c_inp_unfold.clone().detach())

    g_out = torch.randn_like(t_out)
    t_out.backward(g_out)
    c_out.backward(g_out)
    c_out_scatter.backward(g_out)
    c_out_unfold.backward(g_out)
    assert torch.allclose(c_out.data, c_out_scatter.data, atol=1e-5)
    assert torch.allclose(t_out.data, c_out_scatter.data, atol=1e-5)
    assert torch.allclose(t_out.data, c_out_unfold.data, atol=1e-5)
    assert torch.allclose(c_out_qdq_unfold.data, c_out_unfold.data, atol=1e-5)
    assert torch.allclose(c_out_qdq_scatter.data, c_out_scatter.data, atol=1e-5)

    assert torch.allclose(c_inp.grad, c_inp_scatter.grad, atol=1e-5)
    assert torch.allclose(t_inp.grad, c_inp_scatter.grad, atol=1e-5)
    assert torch.allclose(t_inp.grad, c_inp_unfold.grad, atol=1e-5)

    # config rule corresponds to the Conv1d config in the BASIC mode
    config_rule = dict(
        input_formats=[format.BFP16_64],
        weight_format=format.BFP16_64,
        bias_format=format.BFP32_1,
        output_formats=[format.FLOAT16],
    )
    dmx_module.configure(config_rule)
    dmx_module_scatter.configure(config_rule)
    dmx_module_unfold.configure(config_rule)
    c_out = dmx_module(c_inp)
    c_out_scatter = dmx_module_scatter(c_inp_scatter)
    c_out_unfold = dmx_module_unfold(c_inp_unfold)
    g_out = torch.randn_like(t_out)
    c_out.backward(g_out)
    c_out_scatter.backward(g_out)
    c_out_unfold.backward(g_out)
    assert torch.allclose(c_out.data, c_out_scatter.data, atol=1e-5)
    assert torch.allclose(c_inp.grad, c_inp_scatter.grad, atol=1e-5)
    assert torch.allclose(c_out.data, c_out_unfold.data, atol=1e-5)
    assert torch.allclose(c_inp.grad, c_inp_unfold.grad, atol=1e-5)


@pytest.mark.parametrize("bias", (False, ))
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride",
    (
        (3, 768, 16, 16),
        (3, 768, 32, 32),
        (3, 1024, 14, 14),
    ),
)
@pytest.mark.parametrize(
    "image_size",
    (
        (224, 224),
    ),
)
@pytest.mark.parametrize(
    "batch_size",
    (
        1,
    ),
)
def test_conv2d_gather(
    batch_size,
    image_size,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    bias,
):
    torch_module = eval(f"torch.nn.Conv2d")(
        in_channels, out_channels, kernel_size, stride=stride, bias=bias, device=device
    )
    dmx_module = eval(f"dmxnn.Conv2d")(
        in_channels, out_channels, kernel_size, stride=stride, bias=bias, device=device
    )
    dmx_module_gather = eval(f"Conv2dGather")(
        in_channels, out_channels, kernel_size, stride=stride, bias=bias, device=device
    )
    dmx_module_unfold = eval(f"Conv2dUnfold")(
        in_channels, out_channels, kernel_size, stride=stride, bias=bias, device=device
    )
    dmx_module.weight.data = torch_module.weight.data
    dmx_module_gather.weight.data = torch_module.weight.data
    dmx_module_unfold.weight.data = torch_module.weight.data

    if bias:
        dmx_module.bias.data = torch_module.bias.data
        dmx_module_gather.bias.data = torch_module.bias.data
        dmx_module_unfold.bias.data = torch_module.bias.data

    t_inp = torch.randn(
        batch_size, in_channels, *image_size, device=device
    ).requires_grad_()
    c_inp = t_inp.clone().detach().requires_grad_()
    c_inp_gather = t_inp.clone().detach().requires_grad_()
    c_inp_unfold = t_inp.clone().detach().requires_grad_()
    t_out = torch_module(t_inp)
    c_out = dmx_module(c_inp)
    c_out_gather = dmx_module_gather(c_inp_gather)
    c_out_unfold = dmx_module_unfold(c_inp_unfold)

    g_out = torch.randn_like(t_out)
    t_out.backward(g_out)
    c_out.backward(g_out)
    c_out_gather.backward(g_out)
    c_out_unfold.backward(g_out)
    assert torch.allclose(c_out.data, c_out_gather.data, atol=1e-5)
    assert torch.allclose(t_out.data, c_out_gather.data, atol=1e-5)
    assert torch.allclose(t_out.data, c_out_unfold.data, atol=1e-5)

    assert torch.allclose(c_inp.grad, c_inp_gather.grad, atol=1e-5)
    assert torch.allclose(t_inp.grad, c_inp_gather.grad, atol=1e-5)
    assert torch.allclose(t_inp.grad, c_inp_unfold.grad, atol=1e-5)

    # config rule corresponds to the Conv2d config in the BASIC mode
    config_rule = dict(
        input_formats=[format.BFP16_64],
        weight_format=format.BFP16_64,
        bias_format=format.BFP32_1,
        output_formats=[format.FLOAT16],
    )
    dmx_module.configure(config_rule)
    dmx_module_gather.configure(config_rule)
    dmx_module_unfold.configure(config_rule)
    c_out = dmx_module(c_inp)
    c_out_gather = dmx_module_gather(c_inp_gather)
    c_out_unfold = dmx_module_unfold(c_inp_unfold)
    g_out = torch.randn_like(t_out)
    c_out.backward(g_out)
    c_out_gather.backward(g_out)
    c_out_unfold.backward(g_out)
    assert torch.allclose(c_out.data, c_out_gather.data, atol=1e-5)
    assert torch.allclose(c_inp.grad, c_inp_gather.grad, atol=1e-5)
    assert torch.allclose(c_out.data, c_out_unfold.data, atol=1e-5)
    assert torch.allclose(c_inp.grad, c_inp_unfold.grad, atol=1e-5)