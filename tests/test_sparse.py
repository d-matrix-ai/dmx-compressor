import pytest

import torch
from mltools import corsair
from mltools.sparse import Sparsify, Sparseness

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


@pytest.mark.parametrize(
    "tensor_shape",
    (
            (1024, 1024),
            (256, 256, 32, 32),
            (8, 256, 256),
    ),
)
def test_transformation(tensor_shape):
    """Test updating of sparseness and backward mode."""
    sp = Sparsify(tensor_shape).to(device)
    assert repr(sp.sparseness) == "DENSE"
    assert sp.backward_mode == "STE"

    # Old API: directly sets sparseness and backward mode.
    # New API: uses configure.
    # Tests that these two APIs are the same.
    sp.configure(sparseness="BERN", backward_mode="supermask")
    assert repr(sp.sparseness) == "BERN"
    assert sp.backward_mode == "supermask"

    sp.sparseness = Sparseness.from_shorthand("DENSE")
    sp.backward_mode = "STE"
    assert repr(sp.sparseness) == "DENSE"
    assert sp.backward_mode == "STE"

    sp.configure(sparseness="TOPK{0.5}", backward_mode="joint")
    assert repr(sp.sparseness) == "TOPK{0.5}"
    assert sp.backward_mode == "joint"

    sp.sparseness = Sparseness.from_shorthand("BTOPK{4:8,-1}")
    sp.backward_mode = "STE"
    assert repr(sp.sparseness) == "BTOPK{4:8,-1}"
    assert sp.backward_mode == "STE"
