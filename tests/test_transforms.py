#!/usr/bin/env python3

import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn
from mltools import corsair
from mltools.corsair.transform import CorsairTransform
import inspect
import ipdb

RANDOM_SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)

def test_corsair_transform():
    net = Net()
    net = corsair.Model(net)
    gm = torch.fx.symbolic_trace(net.body)
    # ipdb.set_trace()
    transformed : torch.nn.Module = CorsairTransform(gm).transform()
    # ipdb.set_trace()
    # Found that gm was transformed in place to Corsair
    assert (isinstance(gm.linear, corsair.nn.Linear))
    # assert (isinstance(transformed.linear, corsair.nn.Linear))


class Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, input):
        return self.linear(input)
