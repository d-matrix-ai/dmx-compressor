import pytest
import torch
import torch.nn.functional as F
from dmx.compressor import dmx
from dmx.compressor.numerical import Format
from dmx.compressor.functional import ApproximationFunction


@pytest.mark.parametrize(
    "bsz",
    (
        1,
        8,
    ),
)
@pytest.mark.parametrize(
    "seq_len",
    (
        128,
        384,
        1024,
        2048,
    ),
)
@pytest.mark.parametrize(
    "embed_dim",
    (
        128,
        768,
        1024,
    ),
)
@pytest.mark.parametrize("algo", ("fallback", "legacy", "quake3"))
@pytest.mark.parametrize("nform", ("float16",))
@pytest.mark.parametrize("norm", (4,))
@pytest.mark.parametrize("eps", (1e-5, 1e-12, 2.0**-126))
def test_layernorm(bsz, seq_len, embed_dim, algo, nform, norm, eps):
    shape = (bsz, seq_len, embed_dim)
    normalized_shape = (embed_dim,)
    gamma = torch.rand(normalized_shape).requires_grad_()
    beta = torch.randn(normalized_shape).requires_grad_()
    x = torch.randn(*shape) * 3.0 - 1.5

    ln0 = lambda x, gamma, beta: F.layer_norm(
        x, normalized_shape=normalized_shape, weight=gamma, bias=beta, eps=eps
    )
    ln1 = dmx.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=True)
    ln1.weight.data, ln1.bias.data = gamma.data, beta.data
    ln2 = dmx.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=False)
    ln1.transform(
        dict(
            input_format=Format.from_shorthand("SAME"),
            output_format=Format.from_shorthand("SAME"),
            approximation_function=ApproximationFunction.from_shorthand(
                f"LAYERNORM({algo},{norm},{nform})"
            ),
        )
    )
    ln2.transform(
        dict(
            input_format=Format.from_shorthand("SAME"),
            output_format=Format.from_shorthand("SAME"),
            approximation_function=ApproximationFunction.from_shorthand(
                f"LAYERNORM({algo},{norm},{nform})"
            ),
        )
    )
    x10 = x.clone().detach().requires_grad_()
    x1 = x.clone().detach().requires_grad_()
    x20 = x.clone().detach().requires_grad_()
    x2 = x.clone().detach().requires_grad_()

    y10 = ln0(x10, gamma, beta)
    y1 = ln1(x1)
    y20 = ln0(x20, None, None)
    y2 = ln2(x2)

    y1.backward(torch.ones_like(x))
    y10.backward(torch.ones_like(x))
    y20.backward(torch.ones_like(x))
    y2.backward(torch.ones_like(x))

    if algo == "quake3":
        atol = 3e-2
    else:
        atol = 1e-2

    assert y1.shape == y2.shape == y10.shape == y20.shape
    assert torch.allclose(
        ln1.approximation_error, torch.zeros_like(x), rtol=0.0, atol=atol
    )
    assert torch.allclose(
        ln2.approximation_error, torch.zeros_like(x), rtol=0.0, atol=atol
    )
    assert torch.all(x10.grad == x1.grad)
    assert torch.all(gamma.grad == ln1.weight.grad)
    assert torch.all(beta.grad == ln1.bias.grad)
    assert torch.all(x20.grad == x2.grad)
