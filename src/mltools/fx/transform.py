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
