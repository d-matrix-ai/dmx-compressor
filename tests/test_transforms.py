#!/usr/bin/env python3

import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn
from mltools import corsair
from mltools.models import LeNet
from mltools.corsair.transform import cast_input_output_transform
import inspect
import ipdb

RANDOM_SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)

def test_corsair_transform():
    net = Net()

    cnet = CorsairNet()

    cgm = torch.fx.symbolic_trace(cnet)

    gm = cast_input_output_transform(net)
    assert(cgm.code == gm.code)

def downcast(input):
    return input

def upcast(input):
    return input

def test_fakecast_transform():
    net = Net()
    # cnet = FakecastNet()

    # cgm = torch.fx.symbolic_trace(cnet)
    fakecode = '\n\n\ndef forward(self, input):\n    input_1 = input\n    downcast = test_transforms_downcast(input_1);  input_1 = None\n    linear = self.linear(downcast);  downcast = None\n    upcast = test_transforms_upcast(linear);  linear = None\n    return upcast\n    '
    gm = cast_input_output_transform(net, downcast, upcast)
    assert(fakecode == gm.code)

def test_lenet_1hid_corsair_transform():
    net = LeNet([10,10])
    fakecode = '\n\n\ndef forward(self, x : torch.Tensor) -> torch.Tensor:\n    clone = x.clone();  x = None\n    input_layer = self.input_layer(clone);  clone = None\n    clone_1 = input_layer.clone();  input_layer = None\n    clone_2 = clone_1.clone();  clone_1 = None\n    act_func = self.act_func(clone_2);  clone_2 = None\n    clone_3 = act_func.clone();  act_func = None\n    clone_4 = clone_3.clone();  clone_3 = None\n    intermediate_layers_0 = getattr(self.intermediate_layers, "0")(clone_4);  clone_4 = None\n    clone_5 = intermediate_layers_0.clone();  intermediate_layers_0 = None\n    clone_6 = clone_5.clone();  clone_5 = None\n    act_func_1 = self.act_func(clone_6);  clone_6 = None\n    clone_7 = act_func_1.clone();  act_func_1 = None\n    clone_8 = clone_7.clone();  clone_7 = None\n    output_layer = self.output_layer(clone_8);  clone_8 = None\n    clone_9 = output_layer.clone();  output_layer = None\n    return clone_9\n    '
    gm = cast_input_output_transform(net)
    assert(fakecode == gm.code)

def test_lenet_5hid_corsiar_transform():
    net = LeNet([10,10,10,10,10,10])
    fakecode = '\n\n\ndef forward(self, x : torch.Tensor) -> torch.Tensor:\n    clone = x.clone();  x = None\n    input_layer = self.input_layer(clone);  clone = None\n    clone_1 = input_layer.clone();  input_layer = None\n    clone_2 = clone_1.clone();  clone_1 = None\n    act_func = self.act_func(clone_2);  clone_2 = None\n    clone_3 = act_func.clone();  act_func = None\n    clone_4 = clone_3.clone();  clone_3 = None\n    intermediate_layers_0 = getattr(self.intermediate_layers, "0")(clone_4);  clone_4 = None\n    clone_5 = intermediate_layers_0.clone();  intermediate_layers_0 = None\n    clone_6 = clone_5.clone();  clone_5 = None\n    act_func_1 = self.act_func(clone_6);  clone_6 = None\n    clone_7 = act_func_1.clone();  act_func_1 = None\n    clone_8 = clone_7.clone();  clone_7 = None\n    intermediate_layers_1 = getattr(self.intermediate_layers, "1")(clone_8);  clone_8 = None\n    clone_9 = intermediate_layers_1.clone();  intermediate_layers_1 = None\n    clone_10 = clone_9.clone();  clone_9 = None\n    act_func_2 = self.act_func(clone_10);  clone_10 = None\n    clone_11 = act_func_2.clone();  act_func_2 = None\n    clone_12 = clone_11.clone();  clone_11 = None\n    intermediate_layers_2 = getattr(self.intermediate_layers, "2")(clone_12);  clone_12 = None\n    clone_13 = intermediate_layers_2.clone();  intermediate_layers_2 = None\n    clone_14 = clone_13.clone();  clone_13 = None\n    act_func_3 = self.act_func(clone_14);  clone_14 = None\n    clone_15 = act_func_3.clone();  act_func_3 = None\n    clone_16 = clone_15.clone();  clone_15 = None\n    intermediate_layers_3 = getattr(self.intermediate_layers, "3")(clone_16);  clone_16 = None\n    clone_17 = intermediate_layers_3.clone();  intermediate_layers_3 = None\n    clone_18 = clone_17.clone();  clone_17 = None\n    act_func_4 = self.act_func(clone_18);  clone_18 = None\n    clone_19 = act_func_4.clone();  act_func_4 = None\n    clone_20 = clone_19.clone();  clone_19 = None\n    intermediate_layers_4 = getattr(self.intermediate_layers, "4")(clone_20);  clone_20 = None\n    clone_21 = intermediate_layers_4.clone();  intermediate_layers_4 = None\n    clone_22 = clone_21.clone();  clone_21 = None\n    act_func_5 = self.act_func(clone_22);  clone_22 = None\n    clone_23 = act_func_5.clone();  act_func_5 = None\n    clone_24 = clone_23.clone();  clone_23 = None\n    output_layer = self.output_layer(clone_24);  clone_24 = None\n    clone_25 = output_layer.clone();  output_layer = None\n    return clone_25\n    '
    gm = cast_input_output_transform(net)
    assert(fakecode == gm.code)

def test_lenet_1hid_fakecast_transform():
    net = LeNet([10,10])
    fakecode = '\n\n\ndef forward(self, x : torch.Tensor) -> torch.Tensor:\n    downcast = test_transforms_downcast(x);  x = None\n    input_layer = self.input_layer(downcast);  downcast = None\n    upcast = test_transforms_upcast(input_layer);  input_layer = None\n    downcast_1 = test_transforms_downcast(upcast);  upcast = None\n    act_func = self.act_func(downcast_1);  downcast_1 = None\n    upcast_1 = test_transforms_upcast(act_func);  act_func = None\n    downcast_2 = test_transforms_downcast(upcast_1);  upcast_1 = None\n    intermediate_layers_0 = getattr(self.intermediate_layers, "0")(downcast_2);  downcast_2 = None\n    upcast_2 = test_transforms_upcast(intermediate_layers_0);  intermediate_layers_0 = None\n    downcast_3 = test_transforms_downcast(upcast_2);  upcast_2 = None\n    act_func_1 = self.act_func(downcast_3);  downcast_3 = None\n    upcast_3 = test_transforms_upcast(act_func_1);  act_func_1 = None\n    downcast_4 = test_transforms_downcast(upcast_3);  upcast_3 = None\n    output_layer = self.output_layer(downcast_4);  downcast_4 = None\n    upcast_4 = test_transforms_upcast(output_layer);  output_layer = None\n    return upcast_4\n    '
    gm = cast_input_output_transform(net,downcast,upcast)
    assert(fakecode == gm.code)

def test_lenet_5hid_fakecast_transform():
    net = LeNet([10,10,10,10,10,10])
    fakecode = '\n\n\ndef forward(self, x : torch.Tensor) -> torch.Tensor:\n    downcast = test_transforms_downcast(x);  x = None\n    input_layer = self.input_layer(downcast);  downcast = None\n    upcast = test_transforms_upcast(input_layer);  input_layer = None\n    downcast_1 = test_transforms_downcast(upcast);  upcast = None\n    act_func = self.act_func(downcast_1);  downcast_1 = None\n    upcast_1 = test_transforms_upcast(act_func);  act_func = None\n    downcast_2 = test_transforms_downcast(upcast_1);  upcast_1 = None\n    intermediate_layers_0 = getattr(self.intermediate_layers, "0")(downcast_2);  downcast_2 = None\n    upcast_2 = test_transforms_upcast(intermediate_layers_0);  intermediate_layers_0 = None\n    downcast_3 = test_transforms_downcast(upcast_2);  upcast_2 = None\n    act_func_1 = self.act_func(downcast_3);  downcast_3 = None\n    upcast_3 = test_transforms_upcast(act_func_1);  act_func_1 = None\n    downcast_4 = test_transforms_downcast(upcast_3);  upcast_3 = None\n    intermediate_layers_1 = getattr(self.intermediate_layers, "1")(downcast_4);  downcast_4 = None\n    upcast_4 = test_transforms_upcast(intermediate_layers_1);  intermediate_layers_1 = None\n    downcast_5 = test_transforms_downcast(upcast_4);  upcast_4 = None\n    act_func_2 = self.act_func(downcast_5);  downcast_5 = None\n    upcast_5 = test_transforms_upcast(act_func_2);  act_func_2 = None\n    downcast_6 = test_transforms_downcast(upcast_5);  upcast_5 = None\n    intermediate_layers_2 = getattr(self.intermediate_layers, "2")(downcast_6);  downcast_6 = None\n    upcast_6 = test_transforms_upcast(intermediate_layers_2);  intermediate_layers_2 = None\n    downcast_7 = test_transforms_downcast(upcast_6);  upcast_6 = None\n    act_func_3 = self.act_func(downcast_7);  downcast_7 = None\n    upcast_7 = test_transforms_upcast(act_func_3);  act_func_3 = None\n    downcast_8 = test_transforms_downcast(upcast_7);  upcast_7 = None\n    intermediate_layers_3 = getattr(self.intermediate_layers, "3")(downcast_8);  downcast_8 = None\n    upcast_8 = test_transforms_upcast(intermediate_layers_3);  intermediate_layers_3 = None\n    downcast_9 = test_transforms_downcast(upcast_8);  upcast_8 = None\n    act_func_4 = self.act_func(downcast_9);  downcast_9 = None\n    upcast_9 = test_transforms_upcast(act_func_4);  act_func_4 = None\n    downcast_10 = test_transforms_downcast(upcast_9);  upcast_9 = None\n    intermediate_layers_4 = getattr(self.intermediate_layers, "4")(downcast_10);  downcast_10 = None\n    upcast_10 = test_transforms_upcast(intermediate_layers_4);  intermediate_layers_4 = None\n    downcast_11 = test_transforms_downcast(upcast_10);  upcast_10 = None\n    act_func_5 = self.act_func(downcast_11);  downcast_11 = None\n    upcast_11 = test_transforms_upcast(act_func_5);  act_func_5 = None\n    downcast_12 = test_transforms_downcast(upcast_11);  upcast_11 = None\n    output_layer = self.output_layer(downcast_12);  downcast_12 = None\n    upcast_12 = test_transforms_upcast(output_layer);  output_layer = None\n    return upcast_12\n    '
    gm = cast_input_output_transform(net,downcast,upcast)
    assert(fakecode == gm.code)



class Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, input):
        return self.linear(input)

class CorsairNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, input):
        input_1 = input
        clone = input_1.clone();
        linear = self.linear(clone);
        clone_1 = linear.clone();
        return clone_1

class FakecastNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, input):
        input_1 = input
        downcast1 = downcast(input_1);
        linear = self.linear(downcast1);
        upcast1 = upcast(linear);
        return upcast1
