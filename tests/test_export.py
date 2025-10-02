import torch

import torch.nn.functional as F
from dmx.compressor.modeling.nn import ScaledDotProductAttention
from dmx.compressor.modeling import DmxModel
from torch import nn
import dmx.compressor.modeling.nn as dmxnn

from packaging import version
import pytest

if not (
    "dev" in torch.__version__
    or version.parse(torch.__version__) >= version.parse("2.9.0")
):
    pytest.skip("Requires nightly PyTorch or >= 2.9.0", allow_module_level=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(20, 5)
        self.sub = Submod()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.sub(x)
        x = self.layer2(x)
        if x.shape[0] == 1:
            x = x * 0
        return x


class Submod(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lin = nn.Linear(20, 20)
        self.norm = Norm()

    def forward(self, x):
        return self.lin(self.norm(x))


class Norm(nn.Module):
    """Normalization module"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_normalize(x)


def custom_normalize(x: torch.Tensor) -> torch.Tensor:
    """Custom normalization function"""
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std.add(1e-6) + 1e-6)


def test_export_simple_mlp():
    model1 = SimpleMLP().to(device)
    model2 = SimpleMLP().to(device)
    model2.load_state_dict(model1.state_dict())
    inps = torch.randn(3, 10).to(device)
    inps1 = torch.randn(2, 10).to(device)
    with torch.no_grad():
        out0 = model1(inps)
        out10 = model1(inps1)

        model1 = DmxModel.from_torch(model1, export=False)
        model2 = DmxModel.from_torch(model2, export=True)

        out1 = model1(inps)
        out2 = model2(inps)
        out11 = model1(inps1)
        out12 = model2(inps1)

        assert torch.allclose(
            out1, out2
        ), "mismatch in unquantized fx and unquantized export!"
        assert torch.allclose(out0, out1), "mismatch in unquantized fx and baseline!"
        assert torch.allclose(
            out10, out11
        ), "mismatch in unquantized fx and unquantized export!"
        assert torch.allclose(out11, out12), "mismatch in unquantized fx and baseline!"

        for n, m in model1.named_modules():
            if isinstance(m, dmxnn.Linear):
                m.weight_cast.set_format("XP[2,0](CSN)")

        for n, m in model2.named_modules():
            if isinstance(m, dmxnn.Linear):
                m.weight_cast.set_format("XP[2,0](CSN)")

        out3 = model1(inps)
        out4 = model2(inps)
        assert not torch.allclose(
            out3, out0
        ), "quantized fx should be different from baseline!"
        assert torch.allclose(
            out3, out4
        ), "mismatch in quantized fx and quantized export!"

        out13 = model1(inps1)
        out14 = model2(inps1)
        assert not torch.allclose(
            out13, out10
        ), "quantized fx should be different from baseline!"
        assert torch.allclose(
            out13, out14
        ), "mismatch in quantized fx and quantized export!"


def test_export_simple_mlp_backward():
    model1 = SimpleMLP().to(device)
    model2 = SimpleMLP().to(device)
    model0 = SimpleMLP().to(device)
    model1.load_state_dict(model0.state_dict())
    model2.load_state_dict(model0.state_dict())
    inps = torch.randn(3, 10).to(device)

    out0 = model0(inps)

    model1 = DmxModel.from_torch(model1, export=False)
    model2 = DmxModel.from_torch(model2, export=True)

    with torch.enable_grad():
        out1 = model1(inps)
        out2 = model2(inps)

        ref = torch.rand(out0.shape).to(device)

        from torch.nn import MSELoss

        model0.zero_grad()
        model1.zero_grad()
        model2.zero_grad()
        loss0 = MSELoss()(out0, ref)
        loss1 = MSELoss()(out1, ref)
        loss2 = MSELoss()(out2, ref)
        loss0.backward()
        loss1.backward()
        loss2.backward()

        assert torch.allclose(
            model1.layer1.weight.grad, model2.layer1.weight.grad
        ), "mismatch in gradients for unquantized fx and unquantized export!"
        assert torch.allclose(
            model1.layer1.weight.grad, model0.layer1.weight.grad
        ), "mismatch in gradients for unquantized fx and baseline!"

        model1.to_basic_mode()
        model2.to_basic_mode()

        model1.zero_grad()
        model2.zero_grad()
        out1_quant = model1(inps)
        out2_quant = model2(inps)
        loss1_quant = MSELoss()(out1_quant, ref)
        loss2_quant = MSELoss()(out2_quant, ref)
        loss1_quant.backward()
        loss2_quant.backward()

    assert torch.allclose(
        model1.layer1.weight.grad, model2.layer1.weight.grad, atol=1e-4
    ), "mismatch in gradients for quantized fx and quantized export!"
    assert not torch.allclose(
        model1.layer1.weight.grad, model0.layer1.weight.grad
    ), "gradients for quantized fx and baseline should be different!"


class CustomMod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(512, 512)
        self.value = torch.nn.Linear(512, 512)
        self.query = torch.nn.Linear(512, 512)

    def forward(self, x, s=None):
        key = self.key(x)
        value = self.value(x)
        query = self.query(x)
        out = F.scaled_dot_product_attention(query, key, value, scale=s)
        return out


class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mods = torch.nn.ModuleList([CustomMod() for _ in range(5)])

    def forward(self, x, s=None):
        for mod in self.mods:
            x = mod(x, s)
        return x


def test_sdpa():
    model1 = CustomModel().to(device)
    model2 = CustomModel().to(device)
    model2.load_state_dict(model1.state_dict())
    x = torch.rand(1, 512).to(device)
    with torch.no_grad():
        y0 = model1(x)
    model1 = DmxModel.from_torch(model1, export=True)
    model2 = DmxModel.from_torch(model2, export=False)
    with torch.no_grad():
        y1 = model1(x)
        y2 = model2(x)
    assert torch.allclose(y0, y1), "mismatch in unquantized fx and baseline!"
    assert torch.allclose(y1, y2), "mismatch in unquantized fx and unquantized export!"
    model1.to_basic_mode()
    model2.to_basic_mode()
    with torch.no_grad():
        y1_quant = model1(x)
        y2_quant = model2(x)
    assert torch.allclose(
        y1_quant, y2_quant
    ), "mismatch in quantized fx and quantized export!"
    assert not torch.allclose(
        y0, y1_quant
    ), "quantized fx should be different from baseline!"


def test_advance_recipe():
    model1 = CustomModel().to(device)
    model2 = CustomModel().to(device)
    model2.load_state_dict(model1.state_dict())
    x = torch.rand(1, 512).to(device)
    with torch.no_grad():
        out0 = model1(x)
    model1 = DmxModel.from_torch(model1, export=True)
    model2 = DmxModel.from_torch(model2, export=False)

    # Testing Smoothquant
    from dmx.compressor import nn

    from dmx.compressor.advanced_recipe import (
        DmxModuleSmoothQuantHyperparams,
        DmxSmoothQuantRecipe,
    )

    def hp_gen(_model) -> dict:
        return {
            _m: DmxModuleSmoothQuantHyperparams(
                migration_strength=0.25,
                fuse_to_weight=True,
            )
            for _, _m in _model.named_dmx_modules()
            if isinstance(_m, (nn.Linear,))
        }

    with DmxSmoothQuantRecipe(hp_gen).applied_to(model1):
        model1(x)

    # -------------------------------------------------------------------------------
    out1 = model1(x)
    out2 = model2(x)
    assert torch.allclose(
        out1, out2
    ), "mismatch in unquantized fx and unquantized export with smoothquant"
    assert torch.allclose(
        out0, out2
    ), "mismatch in baseline and unquantized fx with smoothquant"

    # Test quant without calibration
    from dmx.compressor import DmxConfigRule, format, nn

    DmxConfigRule(
        module_types=(nn.Linear,),
        module_config=dict(
            input_formats=[format.INT8],
            weight_format=format.INT4,
        ),
    ).apply_to(model1)

    DmxConfigRule(
        module_types=(nn.ActActMatMul,),
        module_config=dict(
            input_formats=[format.INT8, format.INT8],
        ),
    ).apply_to(model1)

    # -------------------------------------------------------------------------------
    out1 = model1(x)

    DmxConfigRule(
        module_types=(nn.Linear,),
        module_config=dict(
            input_formats=[format.INT8],
            weight_format=format.INT4,
        ),
    ).apply_to(model2)

    DmxConfigRule(
        module_types=(nn.ActActMatMul,),
        module_config=dict(
            input_formats=[format.INT8, format.INT8],
        ),
    ).apply_to(model2)

    # -------------------------------------------------------------------------------
    out2 = model2(x)
    assert torch.allclose(
        out1, out2
    ), "mismatch in quantized fx and quantized export with smoothquant without calibration"

    # Test quant with calibration
    from dmx.compressor.advanced_recipe import (
        DmxQuantizerCalibrationHyperparams,
        DmxModuleQuantizerCalibrationHyperparams,
        DmxQuantizerCalibrationRecipe,
    )

    def hp_gen(_model) -> dict:
        aw_hp = {
            _m: DmxModuleQuantizerCalibrationHyperparams(
                inputs={"input_cast": DmxQuantizerCalibrationHyperparams()},
                weight=DmxQuantizerCalibrationHyperparams(),
            )
            for _, _m in _model.named_dmx_modules()
            if isinstance(_m, (nn.Linear,))
        }
        aa_hp = {
            _m: DmxModuleQuantizerCalibrationHyperparams(
                inputs={
                    "input_cast": DmxQuantizerCalibrationHyperparams(),
                    "multiplier_cast": DmxQuantizerCalibrationHyperparams(),
                },
            )
            for _, _m in _model.named_dmx_modules()
            if isinstance(_m, nn.ActActMatMul)
        }
        return aw_hp | aa_hp

    with DmxQuantizerCalibrationRecipe(hp_gen).applied_to(model1):
        model1(x)
    # -------------------------------------------------------------------------------
    out1 = model1(x)
    with DmxQuantizerCalibrationRecipe(hp_gen).applied_to(model2):
        model2(x)
    # -------------------------------------------------------------------------------
    out2 = model2(x)
    assert torch.allclose(
        out1, out2
    ), "mismatch in quantized fx and quantized export with smoothquant and calibration"

    # Test GPTQ
    from dmx.compressor.advanced_recipe import (
        DmxModuleGPTQHyperparams,
        DmxGPTQRecipe,
    )

    def hp_gen(_model) -> dict:
        return {
            _m: DmxModuleGPTQHyperparams()
            for _, _m in _model.named_dmx_modules()
            if isinstance(_m, (nn.Linear, nn.Conv2d))
        }

    with DmxGPTQRecipe(hp_gen).applied_to(model1):
        model1(x)
    # -------------------------------------------------------------------------------
    out1 = model1(x)

    with DmxGPTQRecipe(hp_gen).applied_to(model2):
        model2(x)
    out2 = model2(x)
    assert torch.allclose(
        out1, out2
    ), "mismatch in quantized fx and quantized export after GPTQ with calibration and smoothquant"


def test_submod_forward():
    model1 = CustomModel().to(device)
    model2 = CustomModel().to(device)
    model2.load_state_dict(model1.state_dict())
    x = torch.rand(1, 512).to(device)
    with torch.no_grad():
        y0 = model1.mods[2](x)
    model1 = DmxModel.from_torch(model1, export=True)
    model2 = DmxModel.from_torch(model2, export=False)
    with torch.no_grad():
        model1(x)
        model2(x)
        y1 = model1.mods[2](x)
        y2 = model2.mods[2](x)
    assert torch.allclose(y0, y1), "mismatch in unquantized fx and baseline!"
    assert torch.allclose(y1, y2), "mismatch in unquantized fx and unquantized export!"
    model1.to_basic_mode()
    model2.to_basic_mode()
    with torch.no_grad():
        y1_quant = model1.mods[2](x)
        y2_quant = model2.mods[2](x)
    assert torch.allclose(
        y1_quant, y2_quant
    ), "mismatch in quantized fx and quantized export!"
    assert not torch.allclose(
        y0, y1_quant
    ), "quantized fx should be different from baseline!"
