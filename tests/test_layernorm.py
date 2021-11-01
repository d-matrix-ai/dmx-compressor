import pytest
import torch
import torch.nn.functional as F
from mltools import corsair


@pytest.mark.parametrize(
    "bsz",
    (
        1,
        16,
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
@pytest.mark.parametrize("algo", ("quake3",))
@pytest.mark.parametrize("nform", ("float16",))
@pytest.mark.parametrize("eps", (1e-5,))
def test_layernorm(bsz, seq_len, embed_dim, algo, nform, eps):
    shape = (bsz, seq_len, embed_dim)
    normalized_shape = (embed_dim,)
    gamma = torch.rand(normalized_shape).requires_grad_()
    beta = torch.randn(normalized_shape).requires_grad_()
    x = torch.randn(*shape) * 3.0 - 1.5

    ln0 = lambda x, gamma, beta: F.layer_norm(
        x, normalized_shape=normalized_shape, weight=gamma, bias=beta, eps=eps
    )
    ln1 = corsair.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=True)
    ln1.weight.data, ln1.bias.data = gamma.data, beta.data
    ln2 = corsair.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=False)
    ln1._transform(
        dict(
            input_format="SAME",
            output_format="SAME",
            approximation_function=f"LAYERNORM({algo},{nform})",
        )
    )
    ln2._transform(
        dict(
            input_format="SAME",
            output_format="SAME",
            approximation_function=f"LAYERNORM({algo},{nform})",
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

    assert y1.shape == y2.shape == y10.shape == y20.shape
    assert torch.allclose(y1, y10, rtol=0.0, atol=3e-2)
    assert torch.allclose(y2, y20, rtol=0.0, atol=3e-2)
    assert torch.allclose(
        ln1.approximation_error, torch.zeros_like(x), rtol=0.0, atol=3e-2
    )
    assert torch.allclose(
        ln2.approximation_error, torch.zeros_like(x), rtol=0.0, atol=3e-2
    )
    assert torch.all(x10.grad == x1.grad)
    assert torch.all(gamma.grad == ln1.weight.grad)
    assert torch.all(beta.grad == ln1.bias.grad)
    assert torch.all(x20.grad == x2.grad)
