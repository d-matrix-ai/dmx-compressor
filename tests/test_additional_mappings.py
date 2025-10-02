#!/usr/bin/env python3

from collections import OrderedDict

import torch
from torch.library import custom_op

from dmx.compressor.modeling import DmxModel, DmxConfigRule
from dmx.compressor.modeling import nn as dmxnn
from dmx.compressor.numerical import CastTo, CastToDict


@custom_op("mylib::my_func", mutates_args=())
def my_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = x + y
    return out


@my_func.register_fake
def _(x, y):
    return x


class DmxSubmod(dmxnn.DmxModule):
    def __init__(self) -> None:
        super().__init__()
        self.input_casts = CastToDict(
            OrderedDict({"x_cast": CastTo(), "y_cast": CastTo()})
        )

    def _forward(self, x, y):
        out = x + y
        return out


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = my_func(x, x)
        return out


def test_additional_mappings():
    model = Model()
    x = torch.ones((1, 1))
    additional_mappings = {"torch.ops.mylib.my_func.default": DmxSubmod}
    dmx_model = DmxModel.from_torch(model, additional_mappings)
    dmx_model(x)

    assert isinstance(dmx_model._gm.my_func_default, DmxSubmod)


def test_no_additional_mappings():
    model = Model()
    x = torch.ones((1, 1))
    dmx_model = DmxModel.from_torch(model)
    dmx_model(x)

    assert hasattr(dmx_model._gm, "my_func_default") == False


def test_no_additional_mappings_export():
    model = Model()
    x = torch.ones((1, 1))
    dmx_model = DmxModel.from_torch(model, export=True)
    dmx_model(x)
    assert hasattr(dmx_model._gm, "my_func") == False


def test_additional_mappings_export():
    model = Model()
    x = torch.ones((1, 1))
    additional_mappings = {"torch.ops.mylib.my_func.default": DmxSubmod}
    dmx_model = DmxModel.from_torch(
        model, export=True, additional_dmx_aware_mappings=additional_mappings
    )
    dmx_model(x)

    assert isinstance(dmx_model._gm.my_func, DmxSubmod)
