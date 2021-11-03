import pytest
import torch
import torch.nn.functional as F
from mltools import corsair


@pytest.mark.parametrize(
    "bsz",
    (
        1,
        8,
    ),
)
@pytest.mark.parametrize(
    "seq_len",
    (384,),
)
@pytest.mark.parametrize(
    "embed_dim",
    (1024,),
)
@pytest.mark.parametrize("algo", ("poly2",))
@pytest.mark.parametrize("nform", ("float16",))
def test_gelu(bsz, seq_len, embed_dim, algo, nform):
    shape = (bsz, seq_len, embed_dim)
    x = torch.randn(shape) * 3.0

    f0 = F.gelu
    f1 = corsair.nn.GELU()
    f1._transform(
        dict(
            input_format="SAME",
            output_format="SAME",
            approximation_function=f"GELU({algo},{nform})",
        )
    )

    x0 = x.clone().detach().requires_grad_()
    x1 = x.clone().detach().requires_grad_()

    y0 = f0(x0)
    y1 = f1(x1)

    y0.backward(torch.ones_like(x))
    y1.backward(torch.ones_like(x))

    assert y0.shape == y1.shape == shape
    assert torch.allclose(
        f1.approximation_error,
        torch.zeros_like(x),
        rtol=0.0,
        atol=2.5e-2,
    )
    assert torch.all(x0.grad == x1.grad)
