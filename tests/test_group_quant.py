import pytest
import torch
from dmx.compressor.modeling import nn as dmxnn
from dmx.compressor import format
from dmx.compressor.numerical.observer import MinMaxObserver, HistogramObserver
from dmx.compressor.numerical import CastTo
from torch import nn

RANDOM_SEED = 0

torch.manual_seed(RANDOM_SEED)


BATCH_SIZE = 2
IN_DIM = 16
OUT_DIM = 8
PERTURB_CHAN = 4
KER_SIZE = 3
IMG_SIZE = 8


def _create_module(cls):
    if cls == dmxnn.Linear:
        return cls(IN_DIM, OUT_DIM)
    elif cls == dmxnn.Conv1d:
        return cls(IN_DIM, 2, KER_SIZE)
    elif cls == dmxnn.Conv2d:
        return cls(IN_DIM, OUT_DIM, KER_SIZE, KER_SIZE)
    else:
        raise ValueError("unsupported module class {cls}")


def _create_test_input(module: torch.nn.Module):
    if isinstance(module, nn.Conv1d):
        x_ref = torch.randn(BATCH_SIZE, IN_DIM, IMG_SIZE)
    elif isinstance(module, nn.Conv2d):
        x_ref = torch.randn(BATCH_SIZE, IN_DIM, IMG_SIZE, IMG_SIZE)
    else:
        x_ref = torch.randn(BATCH_SIZE, IN_DIM)

    return x_ref


def test_block_size_non_factor():
    """
    verify performance of group quant when group size is not a factor of ch_axis
    """
    cast = CastTo(
        format=format.INT4,
        observer=MinMaxObserver,
        group_size=2,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
    )
    cast.enable_observer()
    x = torch.Tensor([[0, 1], [3, 7], [5.1, 8], [10, 14], [0.1, 0.7]])
    y = torch.Tensor([[0, 1], [3, 7], [6, 8], [10, 14], [0.1, 0.7]])
    assert torch.allclose(cast(x), y, rtol=0.0, atol=1e-6)


def test_block_size_non_factor_linear_weight():
    """
    verify performance of group quant when group size is not a factor of ch_axis called from a linear layer
    """
    in_dim = 2
    out_dim = 5
    layer = dmxnn.Linear(in_dim, out_dim)
    layer.weight_cast.set_format(format.INT4)
    layer.weight.data = torch.Tensor([[0, 1], [3, 7], [5.1, 8], [10, 14], [0.1, 0.7]])
    layer.set_weight_calibrator(
        MinMaxObserver, torch.per_tensor_symmetric, group_size=2
    )
    with layer.calibrating_weight(), torch.no_grad():
        layer._weight
    y = torch.Tensor([[0, 1], [3, 7], [6, 8], [10, 14], [0.1, 0.7]])
    assert torch.allclose(layer._weight, y, rtol=0.0, atol=1e-6)


def test_block_size_non_factor_linear_activation():
    """
    verify performance of group quant when group size is not a factor of ch_axis called from a linear layer
    """
    in_dim = 2
    out_dim = 5
    layer = dmxnn.Linear(in_dim, out_dim)
    layer.input_casts.input_cast.set_format(format.INT4)
    x = torch.Tensor([[0, 1], [3, 7], [5.1, 8], [10, 14], [0.1, 0.7]])
    layer.input_casts.input_cast.ch_axis = 0
    layer.set_activation_calibrator(
        {
            "input_cast": {
                "observer_cls": MinMaxObserver,
                "qscheme_to_overload": torch.per_tensor_symmetric,
                "group_size": 2,
            }
        }
    )
    with layer.calibrating_activation(), torch.no_grad():
        layer(x)
    y = torch.Tensor([[0, 1], [3, 7], [6, 8], [10, 14], [0.1, 0.7]])
    assert torch.allclose(layer.input_casts.input_cast(x), y, rtol=0.0, atol=1e-6)


def test_hypernet_linear():
    """
    Testing that applying hypernet on a column of weight matrix is equivalent to the corresponding column of the hypernet on entire weight matrix
    """
    in_dim = 2
    out_dim = 5
    layer = dmxnn.Linear(in_dim, out_dim)
    layer.weight_cast.set_format(format.INT4)
    layer.set_weight_calibrator(
        MinMaxObserver, torch.per_tensor_symmetric, group_size=2
    )
    with layer.calibrating_weight(), torch.no_grad():
        layer._weight
    per_col_cast = torch.zeros((out_dim, in_dim))
    for i in range(in_dim):
        per_col_cast[:, i] = layer.weight_hypernet(layer.weight[:, i])
    assert torch.allclose(per_col_cast, layer._weight, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize(
    "module_cls",
    (
        dmxnn.Linear,
        dmxnn.Conv1d,
        dmxnn.Conv2d,
    ),
)
@pytest.mark.parametrize("observer", (MinMaxObserver, HistogramObserver))
@pytest.mark.parametrize(
    "qscheme", (torch.per_tensor_affine, torch.per_tensor_symmetric)
)
@pytest.mark.parametrize("format", (format.INT4, format.INT8))
def test_per_tensor_equivalence_weight(module_cls, observer, qscheme, format):
    """
    verify that per group is equivalent to per tensor when group size is set to size of ch_axis
    """
    module = _create_module(module_cls)
    module_ref = _create_module(module_cls)
    module_ref.weight.data = module.weight.data

    module.weight_cast.set_format(format)
    module_ref.weight_cast.set_format(format)
    module.set_weight_calibrator(
        observer_cls=observer,
        qscheme_to_overload=qscheme,
        group_size=module.weight.shape[module.weight_cast.ch_axis],
    )
    module_ref.set_weight_calibrator(
        observer_cls=observer,
        qscheme_to_overload=qscheme,
    )
    with module.calibrating_weight(), torch.no_grad():
        module._weight
    with module_ref.calibrating_weight(), torch.no_grad():
        module_ref._weight
    assert torch.allclose(module._weight, module_ref._weight, rtol=0.0, atol=1e-8)


@pytest.mark.parametrize(
    "module_cls",
    (
        dmxnn.Linear,
        dmxnn.Conv1d,
        dmxnn.Conv2d,
    ),
)
@pytest.mark.parametrize("observer", (MinMaxObserver,))
@pytest.mark.parametrize(
    "qscheme",
    (
        [torch.per_tensor_affine, torch.per_channel_affine],
        [torch.per_tensor_symmetric, torch.per_channel_symmetric],
    ),
)
@pytest.mark.parametrize("format", (format.INT4, format.INT8))
def test_per_channel_equivalence_weight(module_cls, observer, qscheme, format):
    """
    verify that per group is equivalent to per channel when group size is set to 1
    """
    module = _create_module(module_cls)
    module_ref = _create_module(module_cls)
    module_ref.weight.data = module.weight.data

    module.weight_cast.set_format(format)
    module_ref.weight_cast.set_format(format)
    module.set_weight_calibrator(
        observer_cls=observer,
        qscheme_to_overload=qscheme[0],
        group_size=1,
    )
    module_ref.set_weight_calibrator(
        observer_cls=observer,
        qscheme_to_overload=qscheme[1],
    )
    with module.calibrating_weight(), torch.no_grad():
        module._weight
    with module_ref.calibrating_weight(), torch.no_grad():
        module_ref._weight
    assert torch.allclose(module._weight, module_ref._weight, rtol=0.0, atol=1e-8)


@pytest.mark.parametrize(
    "module_cls",
    (
        dmxnn.Linear,
        dmxnn.Conv1d,
        dmxnn.Conv2d,
    ),
)
@pytest.mark.parametrize("observer", (MinMaxObserver, HistogramObserver))
@pytest.mark.parametrize(
    "qscheme", (torch.per_tensor_affine, torch.per_tensor_symmetric)
)
@pytest.mark.parametrize("format", (format.INT4, format.INT8))
def test_per_tensor_equivalence_activation(module_cls, observer, qscheme, format):
    """
    verify that per group is equivalent to per tensor when group size is set to size of ch_axis
    """
    module = _create_module(module_cls)
    module_ref = _create_module(module_cls)
    x = _create_test_input(
        module,
    )
    module.input_casts.input_cast.set_format(format)
    module_ref.input_casts.input_cast.set_format(format)
    module.set_activation_calibrator(
        {
            "input_cast": {
                "observer_cls": observer,
                "qscheme_to_overload": qscheme,
                "group_size": x.shape[module.input_casts.input_cast.ch_axis],
            }
        }
    )
    module_ref.set_activation_calibrator(
        {
            "input_cast": {
                "observer_cls": observer,
                "qscheme_to_overload": qscheme,
            }
        }
    )

    with module.calibrating_activation(), torch.no_grad():
        module(x)
    with module_ref.calibrating_activation(), torch.no_grad():
        module_ref(x)
    assert torch.allclose(
        module.input_casts.input_cast(x),
        module_ref.input_casts.input_cast(x),
        rtol=0.0,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "module_cls",
    (
        dmxnn.Linear,
        dmxnn.Conv1d,
        dmxnn.Conv2d,
    ),
)
@pytest.mark.parametrize("observer", (MinMaxObserver,))
@pytest.mark.parametrize(
    "qscheme",
    (
        [torch.per_tensor_affine, torch.per_channel_affine],
        [torch.per_tensor_symmetric, torch.per_channel_symmetric],
    ),
)
@pytest.mark.parametrize("format", (format.INT4, format.INT8))
def test_per_channel_equivalence_activation(module_cls, observer, qscheme, format):
    """
    verify that per group is equivalent to per channel when group size is set to 1
    """
    module = _create_module(module_cls)
    module_ref = _create_module(module_cls)

    x = _create_test_input(
        module,
    )

    module.input_casts.input_cast.set_format(format)
    module_ref.input_casts.input_cast.set_format(format)
    module.set_activation_calibrator(
        {
            "input_cast": {
                "observer_cls": observer,
                "qscheme_to_overload": qscheme[0],
                "group_size": 1,
            }
        }
    )
    module_ref.set_activation_calibrator(
        {
            "input_cast": {
                "observer_cls": observer,
                "qscheme_to_overload": qscheme[1],
            }
        }
    )
    with module.calibrating_activation(), torch.no_grad():
        module(x)
    with module_ref.calibrating_activation(), torch.no_grad():
        module_ref(x)
    assert torch.allclose(
        module.input_casts.input_cast(x),
        module_ref.input_casts.input_cast(x),
        rtol=0.0,
        atol=1e-8,
    )
