from platform import python_branch
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
    sm1 = lambda x: F.softmax(x, dim=dim)
    sm2 = corsair.nn.Softmax(dim=dim)
    sm2.approximation_function = "poly2softmax"
    x1 = torch.randn(*shape).requires_grad_()
    x2 = x1.clone().detach().requires_grad_()

    y1 = sm1(x1)
    y2 = sm2(x2)

    y1.backward(torch.ones_like(y1))
    y2.backward(torch.ones_like(y2))

    assert y1.shape == y2.shape
    assert torch.allclose(y2, y1, rtol=1e-2)
    assert torch.allclose(
        sm2.approximation_error, 
        torch.zeros_like(y2), 
        atol=1e-3
    )
    assert torch.all(x1.grad == x2.grad)

if __name__ == "__main__":
    test_softmax(
        bsz=1,
        shape=(8,),
        dim=-1,
    )
