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
        16,
    ),
)
@pytest.mark.parametrize(
    "shape",
    (
        (8,),
        (128,),
        (384,),
        (2048,),
        (
            8,
            8,
        ),
        (
            128,
            128,
        ),
        (
            384,
            384,
        ),
        (
            2048,
            2048,
        ),
    ),
)
@pytest.mark.parametrize(
    "dim",
    (
        -1,
        1,
    ),
)
@pytest.mark.parametrize(
    "algo,nform",
    (
        ("poly2", "int"),
        ("poly2", "float32"),
        ("poly2", "float16"),
        ("poly2", "bfloat16"),
        ("base2", "float16"),
        ("base2", "bfloat16"),
        ("base2quake3", "float16"),
    ),
)
def test_softmax(bsz, shape, dim, algo, nform):
    shape = (bsz,) + shape
    sm1 = lambda x: F.softmax(x, dim=dim)
    sm2 = dmx.nn.Softmax(dim=dim)
    sm2.transform(
        dict(
            input_format=Format.from_shorthand("SAME"),
            output_format=Format.from_shorthand("SAME"),
            approximation_function=ApproximationFunction.from_shorthand(
                f"SOFTMAX({algo},{nform})"
            ),
        )
    )
    x1 = torch.randn(*shape).requires_grad_()
    x2 = x1.clone().detach().requires_grad_()

    y1 = sm1(x1)
    y2 = sm2(x2)

    y1.backward(torch.ones_like(y1))
    y2.backward(torch.ones_like(y2))

    assert y1.shape == y2.shape
    assert torch.allclose(y2, y1, rtol=1e-1)
    assert torch.allclose(sm2.approximation_error, torch.zeros_like(y2), atol=2e-2)
    assert torch.all(x1.grad == x2.grad)
