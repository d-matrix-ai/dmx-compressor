import torch
import torch.nn as nn
from ..numerical import CastTo
from ..fx import QuantTracer, InputOutputTransformer
from torch.fx import GraphModule
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import warnings

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
    
    cast_module_names = ["input_cast", "output_cast", "weight_cast"]
    cast_module_functions = [input_fn, output_fn, weight_fn]
    if approximate_fn:
        cast_module_names.append("approximator")
        cast_module_functions.append(approximate_fn)

    for mn, mf in zip(cast_module_names, cast_module_functions):
        add_successful = gm.add_submodule(mn, mf)
        if not add_successful:
            warnings.warn("Error adding modue {}".format(mn), Warning)
    transformer = InputOutputTransformer(gm,tracer.node_name_to_scope,cfg)
    transformed = transformer.transform()

    return transformed
