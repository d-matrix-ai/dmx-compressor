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

    cnet = CorsairNet()
    cnet = corsair.Model(cnet)

    gm = torch.fx.symbolic_trace(net.body)
    cgm = torch.fx.symbolic_trace(cnet.body)
    # ipdb.set_trace()
    for i in gm.graph.nodes:
        if i.target == 'input':
            gm.graph.inserting_after(i)
            gm.graph.create_node('call_method','clone',args=(i,))
        elif i.target =='output':
            gm.graph.inserting_before(i)
            prev=gm.graph.create_node('call_method','clone',args=(prev,))
            i.args = (prev,)
        else:
            if len(i.args)!=0:
                i.args = (prev,)    
        prev = i
    
    # transformed : torch.nn.Module = CorsairTransform(gm).transform()
    # ipdb.set_trace()
    gm.recompile()
    assert(cgm.code == gm.code)

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
        #self.linear = corsair.nn.Linear(64,64)

    def forward(self, input):
        # return self.linear(input)
        input_1 = input
        clone = input_1.clone();  input_1 = None
        linear = self.linear(clone);  clone = None
        clone_1 = linear.clone();  linear = None
        return clone_1
