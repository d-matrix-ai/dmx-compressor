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
import inspect

import ipdb

RANDOM_SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)

def checkTransform(gm:nn.Module) -> bool:
    if not isinstance(gm,fx.graph_module.GraphModule):
        return True
    nodeList = []
    for i in gm.graph.nodes:
        nodeList.append(i)
    for i in nodeList:
        if i.op=="placeholder":
            if i.next.target!='input_cast':
                return False
        elif i.op == 'output':
            if i.prev.target!='output_cast':
                return False
        elif i.op =='get_attr':
            if i.next.target!='weight_cast':
                return False
        elif i.op=="call_module":
            if not checkTransform(gm.get_submodule(i.target)):
                return False
    return True


def test_corsair_transform():
    net = Net()
    gm_before = fx.symbolic_trace(net)
    assert not checkTransform(gm_before),True
    gm = cast_input_output_transform(net)
    assert checkTransform(gm),True


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
        return x+1

def test_fakecast_transform():
    net = Net()
    gm = cast_input_output_transform(net, downcast(), upcast(),weightCast())
    assert checkTransform(gm),True

def test_lenet_1hid_corsair_transform():
    net = LeNet([10,10])
    gm = cast_input_output_transform(net)
    assert checkTransform(gm),True 

def test_lenet_5hid_corsiar_transform():
    net = LeNet([10,10,10,10,10,10])
    gm = cast_input_output_transform(net)
    assert checkTransform(gm),True 

def test_lenet_1hid_fakecast_transform():
    net = LeNet([10,10])
    gm = cast_input_output_transform(net,downcast(),upcast())
    assert checkTransform(gm),True 

def test_lenet_5hid_fakecast_transform():
    net = LeNet([10,10,10,10,10,10])
    gm = cast_input_output_transform(net,downcast(),upcast())
    assert checkTransform(gm),True 

def test_conv2D_fakecast_transform():
    net = torch.nn.Conv2d(16,8,2)
    gm = cast_input_output_transform(net,downcast(),upcast())
    assert checkTransform(gm),True 

def test_Dropout_transform():
    net = torch.nn.Dropout()
    gm = cast_input_output_transform(net)
    assert checkTransform(gm),True 

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
    net = torch.nn.LayerNorm((1,64))
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
        self.linear = nn.Linear(64, 64)

    def forward(self, input):
        input_1 = input
        clone = input_1.clone()
        linear = self.linear(clone)
        clone_1 = linear.clone()
        return clone_1

class FakecastNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, input):
        input_1 = input
        downcast1 = downcast(input_1)
        linear = self.linear(downcast1)
        upcast1 = upcast(linear)
        return upcast1
