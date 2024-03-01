import pytest
import torch
import torch.nn as nn
from mltools import dmx

RANDOM_SEED = 0

torch.manual_seed(RANDOM_SEED)


BATCH_SIZE = 2
IN_DIM = 16
OUT_DIM = 8
PERTURB_CHAN = 4
KER_SIZE = 3
IMG_SIZE = 8


def _create_module(cls):
    if cls == dmx.nn.Linear:
        return cls(IN_DIM, OUT_DIM)
    elif cls == dmx.nn.Conv1d:
        return cls(IN_DIM, 2, KER_SIZE)
    elif cls == dmx.nn.Conv2d:
        return cls(IN_DIM, OUT_DIM, KER_SIZE, KER_SIZE)
    else:
        raise ValueError("unsupported module class {cls}")


def _create_test_input(module: torch.nn.Module, scaler: float):
    with torch.no_grad():
        if isinstance(module, nn.Conv1d):
            x_ref = torch.randn(BATCH_SIZE, IN_DIM, IMG_SIZE)
        elif isinstance(module, nn.Conv2d):
            x_ref = torch.randn(BATCH_SIZE, IN_DIM, IMG_SIZE, IMG_SIZE)
        else:
            x_ref = torch.randn(BATCH_SIZE, IN_DIM)

        index_shape = [1] * x_ref.ndim
        index_shape[module.ch_axis] = PERTURB_CHAN
        index = torch.arange(PERTURB_CHAN).reshape(index_shape)

        # Perturb the values at the specified indices
        perturb_slice = [slice(None)] * x_ref.ndim
        perturb_slice[module.ch_axis] = index
        x_ref[perturb_slice] *= scaler

    return x_ref


@pytest.mark.parametrize(
    "module_cls",
    (
        dmx.nn.Linear,
        dmx.nn.Conv1d,
        dmx.nn.Conv2d,
    ),
)
@pytest.mark.parametrize("dynamic", (True, False))
@pytest.mark.parametrize("migration_strength", (0.5, 0.0, 1.0))
@pytest.mark.parametrize("pow2", (True, False))
@pytest.mark.parametrize("perturbation_scaler", (1.0, 1e-6, 1e6))
def test_smoothquant(
    module_cls, dynamic, migration_strength, pow2, perturbation_scaler
):
    module = _create_module(module_cls)
    assert module.smoothquant.enabled[0] == 0  # disabled

    module.smoothquant.set_dynamic(dynamic)
    module.smoothquant.set_migration_strength(migration_strength)
    module.smoothquant.set_pow2(pow2)

    x_ref = _create_test_input(module, perturbation_scaler).requires_grad_()
    x = x_ref.clone().detach().requires_grad_()
    y_ref = module(x_ref)
    g_out = torch.randn_like(y_ref)
    y_ref.backward(g_out)
    x_ref_grad = x_ref.grad
    w_ref_grad = module.weight.grad
    w_ref = module.weight.data

    module.zero_grad()
    module.smoothquant.enable()
    if not dynamic:  # static mode needs calibration
        with module.calibrating_smoothquant():
            module(x)

    y = module(x)
    y.backward(g_out)
    x_grad = x.grad
    w_grad = module.weight.grad
    w_scaled = module.smoothquant.scale_weight(module.weight)
    w = torch.div(w_scaled.movedim(module.w_ch_axis, -1), module.smoothquant.scale)
    w = w.movedim(-1, module.w_ch_axis)

    assert torch.allclose(y, y_ref, atol=1e-6, rtol=1e-3)
    assert torch.allclose(x_grad, x_ref_grad, atol=1e-6, rtol=1e-3)
    assert torch.allclose(w_grad, w_ref_grad, atol=1e-6, rtol=1e-3)
    assert torch.allclose(w, w_ref)
