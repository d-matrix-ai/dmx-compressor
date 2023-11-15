import torch
import torch.nn as nn

from mltools.fx.tracer import HFQuantTracer, symbolic_trace, hf_symbolic_trace
from ..fx import QuantTracer, InputOutputTransformer
from mltools.fx.transformer import ConfigurationTransformer, DMXAwareTransformer
from torch.fx import GraphModule
from torch import fx
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


def substitute_transform(
    root: torch.nn.Module,
    concrete_args: Optional[Dict[str, Any]] = None,
    hf: bool = False,
    input_names: Optional[List[str]] = None,
):
    """
    A function that transforms the model by substituting torch.nn.modules and activation functions to dmx.nn.modules.

    Args:
        root (torch.nn.Module): model/module to transform
        concrete_args (Dict[str,Any], optional): concrete arguments to be used for tracer. Defaults to None.
        hf (bool, optional): True if root is a huggingface model. Defaults to False

    Returns:
        transformed model
    """
    if hf:
        # # remove hooks from parallel models as it causes error when traced
        # for key, val in root.__dict__.items():
        #     if "hook" in key:
        #         root.__dict__[key] = None
        # if hasattr(root,'for')
        gm, tracer = hf_symbolic_trace(root, input_names, concrete_args=concrete_args)
    else:
        gm, tracer = symbolic_trace(root, concrete_args)
    transformer = DMXAwareTransformer(gm, tracer.node_name_to_scope)
    transformed = transformer.transform()
    for key, val in root.__dict__.items():
        if key not in transformed.__dict__ and key != "forward":
            transformed.__dict__[key] = val
    breakpoint()
    return transformed


def cast_input_output_transform(
    root: torch.nn.Module,
    tracer: Union[QuantTracer, HFQuantTracer] = QuantTracer(),
    concrete_args: Optional[Dict[str, Any]] = None,
    cfg: Optional[str] = None,
) -> nn.Module:
    """
    A function that transforms the module by adding additional ops, which includes:
    - casting
    - approximator
    - sparsifier
    An optional config file can be passed to specify the formats for the additional ops, dummy formats would be used otherwise.

    Args:
        root (torch.nn.Module): model/module to transform
        tracer (Union[QuantTracer, HFQuantTracer], optional): tracer used for tracing the root. Defaults to QuantTracer.
        concrete_args (Dict[str,Any], optional): concrete arguments to be used for tracer. Defaults to None.
        cfg (Optional[str]): config file for setting the added ops formats. Defaults to None.

    Returns:
        transformed model
    """
    graph = tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    gm = GraphModule(tracer.root, graph, name)
    transformer = InputOutputTransformer(gm, tracer.node_name_to_scope, cfg)
    transformed = transformer.transform()
    transformed.scopeDict = transformer.scopeDict
    return transformed


def configure_transform(gm: torch.fx.GraphModule, scopeDict: dict, cfg: str):
    """
    A function that changes the format of the ops according to the cfg file
    Note:
        Configure_transform will only change existing ops and will not add any additional ops.
        Hence it is recommened to pass in a cfg file for cast_input_output_transform to make sure all necessary ops are added.

    Args:
        gm (torch.fx.GraphModule): Graphmodule to apply changes on
        scopeDict (dict): Dictionary that maps node name to scope
        cfg (str): config file for setting the added ops formats.

    Returns:
        Graphmodule with updated formats
    """
    transformer = ConfigurationTransformer(gm, scopeDict, cfg)
    transformer.transform()
    return gm
