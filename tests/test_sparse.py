import pytest

import torch
from mltools import dmx
from mltools.sparse import Sparsify, Sparseness

RANDOM_SEED = 0

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
        "TOPK{0.5}(U)",
        "TOPK{0.5}(M)",
        "BTOPK{4:8,-1}(U)",
        "BTOPK{4:8,-1}(M)",
        "BTOPK{2:8,-1}(U)",
        "BTOPK{2:8,-1}(M)",
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
        assert isinstance(x.grad, torch.Tensor) and isinstance(
            sp.score.grad, torch.Tensor
        )


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

    sp.configure(
        sparseness="BERN",
        backward_mode="supermask",
        score_func=lambda score, input: score,
    )
    assert repr(sp.sparseness) == "BERN"
    assert sp.backward_mode == "supermask"

    sp.sparseness = Sparseness.from_shorthand("DENSE")
    sp.backward_mode = "STE"
    assert repr(sp.sparseness) == "DENSE"
    assert sp.backward_mode == "STE"

    sp.configure(
        sparseness="TOPK{0.5}(U)",
        backward_mode="joint",
        score_func=lambda score, input: torch.abs(input),
    )
    assert repr(sp.sparseness) == "TOPK{0.5}(U)"
    assert sp.backward_mode == "joint"

    sp.sparseness = Sparseness.from_shorthand("BTOPK{4:8,-1}(U)")
    sp.backward_mode = "STE"
    assert repr(sp.sparseness) == "BTOPK{4:8,-1}(U)"
    assert sp.backward_mode == "STE"
