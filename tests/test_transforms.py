#!/usr/bin/env python3

import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn
from mltools import corsair
from mltools.models import LeNet
from mltools.models import BERTStyleFFN
from mltools.corsair.transform import cast_input_output_transform
import torch.fx as fx

RANDOM_SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)


def checkTransform(gm: nn.Module) -> bool:
    if not isinstance(gm, fx.graph_module.GraphModule):
        return True
    nodeList = []
    for i in gm.graph.nodes:
        nodeList.append(i)
    for i in nodeList:
        if i.op == "placeholder":
            if i.next.target != "input_cast":
                return False
        elif i.op == "output":
            if i.prev.target != "output_cast":
                return False
        elif i.op == "get_attr":
            if i.next.target != "weight_cast":
                return False
        elif i.op == "call_module":
            if not checkTransform(gm.get_submodule(i.target)):
                return False
    return True


def test_corsair_transform():
    net = Net()
    cnet = CorsairNet()
    gm_before = fx.symbolic_trace(net)
    assert not checkTransform(gm_before), True
    gm = cast_input_output_transform(net)
    print(gm.code)
    input = torch.rand(1, 64)

    # Initialize bias and weight
    weight = torch.rand(64, 64)
    bias = torch.zeros(1, 64)
    cnet.linear.bias.data = bias
    gm.linear.bias.data = bias
    cnet.linear.weight.data = weight
    gm.linear.weight.data = weight

    coutput = cnet(input)
    output = gm(input)
    assert (coutput - output).abs().sum() == 0, True
    assert checkTransform(gm), True

@pytest.mark.skip()
def test_double_transform():
    net = Net()
    with pytest.raises(Warning):
        gm = cast_input_output_transform(net)
        gm2t = cast_input_output_transform(gm)

class downcast(nn.Module):
    def __init__(self, format="SAME", dump_to=None):
        super().__init__()

    def forward(self, x):
        return x


class upcast(nn.Module):
    def __init__(self, format="SAME", dump_to=None):
        super().__init__()

    def forward(self, x):
        return x


class weightCast(nn.Module):
    def __init__(self, format="SAME", dump_to=None):
        super().__init__()

    def forward(self, x):
        return x


def test_fakecast_transform():
    net = Net()
    gm = cast_input_output_transform(net, downcast(), upcast(), weightCast())
    assert checkTransform(gm), True


@pytest.mark.parametrize(
    "layers",
    (
        [10, 10],
        [10, 10, 10, 10, 10, 10],
    ),
)
def test_lenet_1hid_corsair_transform(layers):
    net = LeNet(layers)
    gm = cast_input_output_transform(net)
    print(gm.code)
    assert checkTransform(gm), True


@pytest.mark.parametrize(
    "layers",
    (
        [10, 10],
        [10, 10, 10, 10, 10, 10],
    ),
)
def test_lenet_fakecast_transform(layers):
    net = LeNet(layers)
    gm = cast_input_output_transform(net, downcast(), upcast())
    assert checkTransform(gm), True


def test_conv2D_fakecast_transform():
    net = torch.nn.Conv2d(16, 8, 2)
    cnet = corsair.nn.Conv2d(16, 8, 2)
    input = torch.rand(1, 16, 16, 16)
    weight = torch.rand(8, 16, 2, 2)
    bias = torch.rand(8)
    net.weight.data = weight
    cnet.weight.data = weight
    net.bias.data = bias
    cnet.bias.data = bias
    coutput = cnet(input)

    gm = cast_input_output_transform(net, downcast(), upcast())
    output = gm(input)
    assert (coutput - output).abs().sum() == 0, True
    assert checkTransform(gm), True


def test_Dropout_transform():
    net = torch.nn.Dropout()
    gm = cast_input_output_transform(net)
    assert checkTransform(gm), True


def test_AdaptiveAvgPool2d_transform():
    net = torch.nn.AdaptiveAvgPool2d(16)
    gm = cast_input_output_transform(net)
    assert checkTransform(gm), True


def test_AvgPool2d_transform():
    net = torch.nn.AvgPool2d(8)
    gm = cast_input_output_transform(net)
    assert checkTransform(gm), True


def test_MaxPool2d_transform():
    net = torch.nn.MaxPool2d(16)
    gm = cast_input_output_transform(net)
    assert checkTransform(gm), True


def test_Softmax_transform():
    net = torch.nn.Softmax()
    gm = cast_input_output_transform(net)
    assert checkTransform(gm), True


def test_LayerNorm_transform():
    net = torch.nn.LayerNorm((1, 64))
    gm = cast_input_output_transform(net)
    assert checkTransform(gm), True


def test_ReLU_transform():
    net = torch.nn.ReLU()
    gm = cast_input_output_transform(net)
    assert checkTransform(gm)


def test_ReLU6_transform():
    net = torch.nn.ReLU6()
    gm = cast_input_output_transform(net)
    assert checkTransform(gm), True


def test_Tanh_transform():
    net = torch.nn.Tanh()
    gm = cast_input_output_transform(net)
    assert checkTransform(gm), True


def test_GELU_transform():
    net = torch.nn.GELU()
    gm = cast_input_output_transform(net)
    assert checkTransform(gm), True


class Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, input):
        return self.linear(input)


class CorsairNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = corsair.nn.Linear(64, 64)

    def forward(self, input):
        return self.linear(input)
