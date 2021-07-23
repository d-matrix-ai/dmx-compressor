import pytest
import torch
import torch.nn.functional as F
import corsair


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
        (32,),
        (128,),
        (384,),
        (1024,),
        (2048,),
        (8, 8,),
        (32, 32,),
        (128, 128,),
        (384, 384,),
        (1024, 1024,),
        (2048, 2048,),
    ),
)
@pytest.mark.parametrize(
    "dim",
    (
        -1,
        1,
    ),
)
def test_softmax(bsz, shape, dim):
    shape = (bsz,) + shape
    x1 = torch.randn(*shape).requires_grad_()
    x2 = x1.clone().detach().requires_grad_()

    ref = F.softmax(x1, dim=dim)
    res = corsair.nn.Softmax(dim=dim)(x2)

    assert ref.shape == res.shape
    assert torch.allclose(ref, res, rtol=1e-2)

    ref.backward(torch.ones_like(ref))
    res.backward(torch.ones_like(res))

    assert torch.all(x1.grad == x2.grad)
