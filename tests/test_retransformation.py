## This test file serves the purpose of testing the inheritance of castTo states after DmxModel retransformation
import torch
from dmx.compressor.modeling import DmxModel, DmxConfigRule
from dmx.compressor.modeling import nn as dmxnn
from copy import deepcopy

torch.manual_seed(0)


class Net(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ln1 = torch.nn.Linear(in_dim, 128)
        self.act = torch.nn.ReLU()
        self.ln2 = torch.nn.Linear(128, out_dim)

    def forward(self, x, y=None):
        x = self.ln1(x)
        x = self.act(x)
        x = self.ln2(x)
        if y is not None:
            return x.T @ y
        return x


in_dim = 10
out_dim = 5
batch = 2

x = torch.rand((batch, in_dim))
y = torch.rand((batch, in_dim))


def test_retransformation_after_calib():
    """
    scales and zero points should be same while observer and fakequant states should be different
    """
    net = Net(in_dim, out_dim)
    model = DmxModel.from_torch(net)
    with torch.no_grad():
        model(x)
    format = "XP[8,0](CSN)"
    rules = (
        DmxConfigRule(
            module_types=(dmxnn.Linear,),
            module_config=dict(
                input_formats=[format],
                weight_format=format,
            ),
        ),
    )
    model.configure(None, *rules)
    target_layers = {
        n: m for n, m in model.named_dmx_modules() if isinstance(m, (dmxnn.Linear,))
    }
    with model.calibrating_activations(
        target_layers.items()
    ), torch.no_grad(), model.calibrating_weights(target_layers.items()):
        model(x)
        target_layers_copy = {n: deepcopy(m) for n, m in target_layers.items()}
    model(y, y)
    # check all scales and zerpoints are the same
    if any(
        (
            m.input_casts.input_cast.scale[0]
            != target_layers_copy[n].input_casts.input_cast.scale[0]
            or m.input_casts.input_cast.zero_point[0]
            != target_layers_copy[n].input_casts.input_cast.zero_point[0]
        )
        for n, m in target_layers.items()
    ):
        assert False, "scale and zero point changed after retransformation!"

    # check all fake_quant_enabled and observer_enabled are different
    if any(
        (
            m.input_casts.input_cast.fake_quant_enabled
            == target_layers_copy[n].input_casts.input_cast.fake_quant_enabled
            or m.input_casts.input_cast.observer_enabled
            == target_layers_copy[n].input_casts.input_cast.observer_enabled
        )
        for n, m in target_layers.items()
    ):
        assert (
            False
        ), "fake_quant_enabled and observer_enabled incorrect after retransformation!"


def test_retransformation_during_calib():
    """
    scales and zero points should be different while observer and fakequant states should be same
    """
    net = Net(in_dim, out_dim)
    model = DmxModel.from_torch(net)
    with torch.no_grad():
        model(x)
    format = "XP[8,0](CSN)"
    rules = (
        DmxConfigRule(
            module_types=(dmxnn.Linear,),
            module_config=dict(
                input_formats=[format],
                weight_format=format,
            ),
        ),
    )
    model.configure(None, *rules)
    target_layers = {
        n: m for n, m in model.named_dmx_modules() if isinstance(m, (dmxnn.Linear,))
    }
    with model.calibrating_activations(
        target_layers.items()
    ), torch.no_grad(), model.calibrating_weights(target_layers.items()):
        model(x)
        target_layers_copy = {n: deepcopy(m) for n, m in target_layers.items()}
        model(y, y)

        # check not all scales and zerpoints are the same
        if all(
            (
                m.input_casts.input_cast.scale[0]
                == target_layers_copy[n].input_casts.input_cast.scale[0]
                and m.input_casts.input_cast.zero_point[0]
                == target_layers_copy[n].input_casts.input_cast.zero_point[0]
            )
            for n, m in target_layers.items()
        ):
            assert (
                False
            ), "scale and zero point did not changed when calibrating after retransformation!"

        # check all fake_quant_enabled and observer_enabled are the same
        if any(
            (
                m.input_casts.input_cast.fake_quant_enabled
                != target_layers_copy[n].input_casts.input_cast.fake_quant_enabled
                or m.input_casts.input_cast.observer_enabled
                != target_layers_copy[n].input_casts.input_cast.observer_enabled
            )
            for n, m in target_layers.items()
        ):
            assert (
                False
            ), "fake_quant_enabled and observer_enabled incorrect after retransformation!"
