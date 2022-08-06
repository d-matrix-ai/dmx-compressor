from turtle import forward
import torch
import torch.nn as nn
from ..numerical import CastTo
from ..fx import QuantTracer, InputOutputTransformer
from torch.fx import GraphModule
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

def cast_input_output_transform(
    root: torch.nn.Module,
    input_fn: nn.Module = CastTo(),
    output_fn: nn.Module = CastTo(),
    weight_fn: nn.Module = CastTo(),
    approximate_fn:nn.Module = None,
    concrete_args: Optional[Dict[str, Any]] = None,
    cfg: Optional[str] = None,
) -> nn.Module:
    """
    A function that transforms the module by adding additional ops, which includes:
    - casting
    - approximator
    - sparsifier
    """
    tracer = QuantTracer()
    graph = tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    gm = GraphModule(tracer.root, graph, name)
    transformer = InputOutputTransformer(gm,tracer.node_name_to_scope,cfg)
    transformed = transformer.transform()
    return transformed

class net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(32,32)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return x

class dNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = net()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return x

class LeNet(nn.Module):
    def __init__(self, hidden_dims, input_dim=784, output_dim=10) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.intermediate_layers = nn.ModuleList(
            [
                nn.Linear(hidden_dims[i], hidden_dims[i + 1])
                for i in range(len(hidden_dims) - 1)
            ]
        )
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.act_func = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.act_func(x)
        for layer in self.intermediate_layers:
            x = layer(x)
            x = self.act_func(x)
        x = self.output_layer(x)
        return x 
