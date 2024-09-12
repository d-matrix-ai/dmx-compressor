import torch

from dmx.compressor.modeling import DmxModel, DmxConfigRule
from dmx.compressor.modeling import nn as dmxnn


class Submod(torch.nn.Module):
    def __init__(self, indim, hiddim, outdim) -> None:
        super().__init__()
        self.lin1 = torch.nn.Linear(indim, hiddim)
        self.act = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(hiddim, outdim)

    def forward(self, x, y):
        out = self.lin1(x + y)
        out = self.act(out)
        out = self.lin2(out)
        return out


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sm1 = Submod(160, 6400, 6400)
        self.act = torch.nn.GELU()
        self.sm2 = Submod(6400, 12800, 10)

    def forward(self, x):
        out = self.sm1(x, x)
        out = self.act(out)
        out = self.sm2(out, out)
        return out


model = DmxModel.from_torch(Model())

inp = torch.rand((1, 160))
out_full_0 = model(inp)


# creating transformed submod forward
DmxModel.create_submod_transform_forward(model, "sm1")
DmxModel.create_submod_transform_forward(model, "sm2")


def test_unquantized():
    out1 = model.sm1.transformed_forward(inp, inp)
    out0 = model.sm1.forward(inp, inp)
    # check sm1 output is same between original model and unquantized dmx model
    assert torch.all(
        out1 == out0
    ), "unquantized submodule produced different output from original submodule, should be same"


def test_quantized():
    bfp16 = "BFP[8|8]{64}(SN)"
    rules = (
        DmxConfigRule(
            module_types=(dmxnn.Linear,),
            module_config=dict(
                input_format=bfp16,
                weight_format=bfp16,
                bias_format=bfp16,
                output_format=bfp16,
            ),
        ),
    )
    model.configure(None, *rules)
    out1 = model.sm1.transformed_forward(inp, inp)
    out0 = model.sm1.forward(inp, inp)
    # check sm1 output is different between original model and quantized dmx model
    assert torch.any(
        out1 != out0
    ), "quantized submodule produced same output as original submodule, should be different"
    out_full = model(inp)
    out_sub = model.sm1.transformed_forward(inp, inp)
    out_sub = torch.nn.GELU()(out_sub)
    out_sub = model.sm2.transformed_forward(out_sub, out_sub)
    # check model output is same between running whole quantized model vs running quantized submodules step-by-step
    assert torch.all(
        out_full == out_sub
    ), "quantized whole model produced different output from chained quantized submodules, should be same"
    # check model output is different between dmx quantized model and original model
    assert torch.any(
        out_full != out_full_0
    ), "quantized whole model produced same output as unquantized whole model, should be different"
