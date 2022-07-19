import pytest

import torch
from mltools import corsair
from mltools.sparse import Sparsify

RANDOM_SEED = 0

corsair.aware()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)


@pytest.mark.parametrize(
    "tensor_shape",
    (
            (1024, 1024),
            (256, 256, 32, 32),
            (8, 256, 256),
    ),
)
@pytest.mark.parametrize(
    "sparseness",
    (
            "TOPK{0.5}",
            "BTOPK{4:8,-1}",
            "BTOPK{2:8,-1}",
            "BERN",
    ),
)
@pytest.mark.parametrize(
    "backward_mode",
    (
            "STE",
            "supermask",
            "joint",
    ),
)
def test_sparsify(tensor_shape, sparseness, backward_mode):
    """Test that `Sparsify` produces a correct sparseness pattern, 
    as evidenced by gradients of weights and scores."""
    sp = Sparsify(tensor_shape, sparseness, backward_mode).to(device)
    x = torch.randn(tensor_shape, requires_grad=True, device=device)
    y = sp(x)
    y.backward(torch.ones_like(y))

    if backward_mode == "STE":
        assert isinstance(x.grad, torch.Tensor) and sp.score.grad is None
    elif backward_mode == "supermask":
        assert x.grad is None and isinstance(sp.score.grad, torch.Tensor)
    elif backward_mode == "joint":
        assert isinstance(x.grad, torch.Tensor) and isinstance(sp.score.grad, torch.Tensor)
