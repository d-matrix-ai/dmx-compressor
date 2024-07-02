#!/usr/bin/env python3

import torch
from torch import fx, nn
import torch.nn.functional as F
from compressor import dmx
from compressor.fx.transform import substitute_transform
from compressor.fx.transformer import dmx_aware_mapping
from compressor.utils.fx.qdq_graph import qdq_attr

import dmir_compiler.custom_ops

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)

def test_qdq_attr():
    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.test_attr = torch.ones((128))
            self.linear = torch.nn.Linear(128,128)

        def forward(self, x):
            return x

    test_module = TestModule()
    gm = fx.symbolic_trace(test_module)
    with gm.graph.inserting_before():
        qdq_attr(gm.graph, 'test_attr', 'SAME')

    assert len(list(filter(lambda n: n.target == "test_attr", gm.graph.nodes))) == 1
    assert len(list(filter(lambda n: n.target == "test_attr_scale", gm.graph.nodes))) == 1
    assert len(list(filter(lambda n: n.target == "test_attr_zero_point", gm.graph.nodes))) == 1
    assert len(list(filter(lambda n: n.target == torch.ops.dmx.quantize, gm.graph.nodes))) == 1
    assert len(list(filter(lambda n: n.target == torch.ops.dmx.dequantize, gm.graph.nodes))) == 1
