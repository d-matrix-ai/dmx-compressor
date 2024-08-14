#!/usr/bin/env python3

import torch
from torch import fx, nn
import torch.nn.functional as F
from dmx.compressor import dmx
from dmx.compressor.fx.transform import substitute_transform
from dmx.compressor.fx.transformer.utils import dmx_aware_mapping

RANDOM_SEED = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)


def check_all_dmx(gm: torch.nn.Module) -> bool:
    """
    Checking that all modules are dmx modules
    """
    assert isinstance(gm, fx.graph_module.GraphModule), True
    for node in gm.graph.nodes:
        if node.op == "call_module":
            module = gm.get_submodule(node.target)
            node_key = type(module).__module__ + "." + type(module).__name__
            if (
                not isinstance(module, dmx.nn.DmxModule)
                and node_key in dmx_aware_mapping
            ):
                return False
    return True


class Lenet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.mp1 = nn.MaxPool2d((2, 2))
        self.mp2 = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.mp1(F.relu(self.conv1(x)))
        x = self.mp2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_lenet5():
    net = Lenet5()
    gm_before = fx.symbolic_trace(net)
    gm = dmx.DmxModel.from_torch(net)
    inp = torch.rand(1, 95, 95)
    output_before = gm_before(inp)
    output = gm(inp)
    assert check_all_dmx(gm._gm), True
    assert (output_before - output).abs().sum() <= 1e-5, True


class MultiInputNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_x = nn.Linear(32, 64)
        self.linear_y = nn.Linear(64, 64)
        self.linear_z = nn.Linear(16, 64)

    def forward(self, x, y, z):
        return self.linear_x(x) + self.linear_y(y) + self.linear_z(z)


def test_multiple_inputs():
    net = MultiInputNet()
    gm_before = fx.symbolic_trace(net)
    gm = dmx.DmxModel.from_torch(net)
    x = torch.rand(8, 32)
    y = torch.rand(8, 64)
    z = torch.rand(8, 16)
    output_before = gm_before(x, y, z)
    output = gm(x, y, z)
    assert check_all_dmx(gm._gm), True
    assert (output_before - output).abs().sum() <= 1e-5, True


class Add(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.iden = nn.Identity()

    def forward(self, x, y):
        return self.iden(x) + y


class ResConnection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(64, 32)
        self.add = Add()

    def forward(self, x):
        return self.add(self.linear_2(self.relu(self.linear_1(x))), x)


def test_res_connection():
    net = ResConnection()
    gm_before = fx.symbolic_trace(net)
    gm = dmx.DmxModel.from_torch(net)
    x = torch.rand(8, 32)
    output_before = gm_before(x)
    output = gm(x)
    assert check_all_dmx(gm._gm), True
    assert (output_before - output).abs().sum() <= 1e-5, True
