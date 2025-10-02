import torch
import pytest

from dmx.compressor.modeling.model import DmxModel, DmxConfigRule
from dmx.compressor.modeling import nn as dmxnn


class Submod(torch.nn.Module):
    def __init__(self, indim, hiddim, outdim) -> None:
        super().__init__()
        self.lin1 = torch.nn.Linear(indim, hiddim)
        self.act = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(hiddim, outdim)

    def forward(self, x, y, relu=True):
        out = self.lin1(x + y)
        if relu:
            out = self.act(out)
        out = self.lin2(out)
        return out


class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sm1 = Submod(160, 6400, 6400)
        self.act = torch.nn.GELU()
        self.sm2 = Submod(6400, 12800, 10)

    def forward(self, x, relu=True):
        out = self.sm1(x, x, relu)
        out = self.act(out)
        out = self.sm2(out, out, relu)
        return out


@pytest.fixture
def setup_model():
    model0 = CustomModel()
    model = CustomModel()
    model.load_state_dict(model0.state_dict())
    model = DmxModel.from_torch(model)
    inp = torch.rand((1, 160))

    return model, model0, inp


@pytest.fixture
def setup_model_export():
    model0 = CustomModel()
    model = CustomModel()
    model.load_state_dict(model0.state_dict())
    model = DmxModel.from_torch(model, export=True)
    inp = torch.rand((1, 160))

    return model, model0, inp


@pytest.fixture(params=["setup_model", "setup_model_export"])
def setup(request):
    return request.getfixturevalue(request.param)


def test_unquantized(setup):
    model, model0, inp = setup
    assert torch.allclose(
        model(inp), model0(inp)
    ), "unquantized model produced different output from original model, should be same"
    assert torch.allclose(
        model0.sm1(inp, inp), model.sm1(inp, inp)
    ), "unquantized submodule produced different output from original submodule, should be same"


def test_quantized(setup):
    model, model0, inp = setup
    model.to_basic_mode()
    basic_output = model(inp)
    ref_output = model0(inp)
    assert not torch.allclose(
        ref_output, basic_output
    ), "quantized model produced same output from original model, should be different"
    assert not torch.allclose(
        model0.sm1(inp, inp), model.sm1(inp, inp)
    ), "quantized submodule produced same output from original submodule, should be different"
    basic_output_from_submod = model.sm1(inp, inp)
    basic_output_from_submod = model.act(basic_output_from_submod)
    basic_output_from_submod = model.sm2(
        basic_output_from_submod, basic_output_from_submod
    )
    torch.allclose(
        basic_output, basic_output_from_submod
    ), "quantized model and running submodule sequentially differ, should be same"


def test_quantize_submod(setup):
    model, model0, inp = setup
    model.sm1.to_basic_mode()
    model(inp)

    assert not torch.allclose(
        model0.sm1(inp, inp, False), model.sm1(inp, inp, False)
    ), "submodule in quantized mode should produce different output from original submodule"
    assert not torch.allclose(
        model(inp), model0(inp)
    ), "model with one quantized submodule should produce different output from original model"


def test_retracing(setup):
    model, model0, inp = setup
    model(inp)
    assert torch.allclose(
        model0.sm1(inp, inp, False), model.sm1(inp, inp, False)
    ), "submodule in unquantized mode should produce same output from original submodule"
    assert torch.allclose(
        model0.sm1(inp, inp, True), model.sm1(inp, inp, True)
    ), " submodule in unquantized mode should produce same output from original submodule"


def test_dmxmod_sharing(setup):
    model, model0, inp = setup
    model(inp)
    model.sm1(inp, inp)
    model.act(inp)
    assert (
        model.sm1._gm.lin1 is model._gm.sm1.lin1
    ), "DmxMod for submodule lin1 should be shared with main gm"
    assert (
        model.act._gm is model._gm.act
    ), "DmxMod activation should be shared with main gm "


def test_forward_switching(setup):
    model, model0, inp = setup
    model.to_basic_mode()
    output_base = model0(inp)
    output_quant = model(inp)
    submod_output_quant = model.sm1(inp, inp)
    submod_output_base = model0.sm1(inp, inp)

    DmxModel.to_old_forward(model)
    assert torch.allclose(
        model(inp), output_base
    ), "to_old_forward should match original baseline model"
    assert torch.allclose(
        model.sm1(inp, inp), submod_output_base
    ), "to_old_forward should match original baseline submodule"

    DmxModel.to_transformed_forward(model)
    assert torch.allclose(
        model(inp), output_quant
    ), "to_transformed_forward should match quantized model"
    assert torch.allclose(
        model.sm1(inp, inp), submod_output_quant
    ), "to_transformed_forward should match quantized submodule"

    DmxModel.to_old_forward(model.sm1)
    assert torch.allclose(
        model.sm1(inp, inp), submod_output_base
    ), "submodule to_old_forward should match original baseline submodule"
    assert not torch.allclose(
        model(inp), output_base
    ), "model should still be in quantized mode"
