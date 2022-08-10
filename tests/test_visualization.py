#!/usr/bin/env python3
import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace
# from mltools.fx.transform import cast_input_output_transform


def make_mlp_and_input():
    model = nn.Sequential()
    model.add_module('W0', nn.Linear(8, 16))
    model.add_module('tanh', nn.Tanh())
    model.add_module('W1', nn.Linear(16, 1))
    x = torch.randn(1, 8)
    return model, x

model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))

x = torch.randn(1,8)

output = make_dot(model(x), params=dict(model.named_parameters()))
output.render(directory='doctest-output', view=True)

# net = nn.Linear(64,64)
# gm = cast_input_output_transform(net)
# x = torch.rand(1,64)
# output = make_dot(model(x), params=dict(model.named_parameters()))
# output.render(directory='doctest-output', view=True)