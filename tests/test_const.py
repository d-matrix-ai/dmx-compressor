#!/usr/bin/env python3

from mltools import corsair
import torch.nn as nn
import torch
from torch import Tensor

corsair.aware()

class TransposeView(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x : Tensor) -> Tensor:
        new_shape = (x.shape[1], x.shape[0])
        x = x.view(new_shape)
        return x


def test_dmir_dump_constant_in_node_args():
    m = TransposeView()
    m = corsair.Model(m)
    input_dim = (128,64)
    sample_input = torch.randn(input_dim)
    output = m(sample_input)
    assert(output.shape == (64,128))
    m.dmir_graph(sample_input=sample_input)
