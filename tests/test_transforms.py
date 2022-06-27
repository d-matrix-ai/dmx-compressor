#!/usr/bin/env python3

import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn
from mltools import corsair
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
