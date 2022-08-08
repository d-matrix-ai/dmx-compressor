#!/usr/bin/env python3

import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn
from mltools import corsair
from mltools.models import LeNet
from mltools.models import BERTStyleFFN
from mltools.fx.transform import cast_input_output_transform
from mltools.sparse import Sparsify
from mltools.approximate import Approximate
import torch.fx as fx
from transformers.pytorch_utils import Conv1D
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention,GPT2MLP,GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
import ipdb

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
            if not i.target+"_cast" in i.next.target:
                return False
        elif i.op == "output":
            if not "output_cast" in i.prev.target:
                return False
        elif i.op == "get_attr" and "weight" in i.target:
            if not ("weight_cast" in i.next.target
                or (("sparsifier" in i.next.target or "approximator" in i.next.target) and "weight_cast" in i.next.next.target) 
                or ("sparsifier" in i.next.target and "approximator" in i.next.next.target and "weight_cast" in i.next.next.next.target)):
                return False
        elif i.op == "get_attr" and "bias" in i.target:
            if not "weight_cast" in i.next.target and not "bias_cast" in i.next.target:
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

    
def test_conv1D():
    net = Conv1D(32,32)
    gm = cast_input_output_transform(net)
 

# def test_gpt2_attention():
#     import ipdb
#     ipdb.set_trace()
#     net = GPT2Model(GPT2Config())
#     gm = cast_input_output_transform(net)
#     ipdb.set_trace()
#     assert checkTransform(gm),True


def test_corsiar_transform_sparsify_approximate():
    net = Net()
    gm = cast_input_output_transform(net,approximate_fn=Approximate(),)
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
def test_lenet_1hid_corsair_transform_without_cfg(layers):
    net = LeNet(layers)
    gm = cast_input_output_transform(net)
    assert checkTransform(gm), True


@pytest.mark.parametrize(
    "layers",
    (
        [10, 10],
    ),
)
def test_lenet_1hid_corsair_transform_with_test_cfg(layers):
    net = LeNet(layers)
    gm = cast_input_output_transform(net,cfg="configs/lenet_test.yaml")
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