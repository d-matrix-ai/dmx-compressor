import torch
from dmx.compressor.modeling import nn as dmxnn


def test_exp():
    net = dmxnn.Exp()
    x = torch.rand(8, 32)
    output_before = torch.exp(x)
    output = net(x)
    assert (output_before - output).abs().sum() <= 1e-5, True
