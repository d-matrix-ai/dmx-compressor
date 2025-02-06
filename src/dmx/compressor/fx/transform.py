import torch
import torch.nn as nn

from dmx.compressor.fx.tracer import HFQuantTracer, hf_symbolic_trace
from ..fx import QuantTracer, InputOutputTransformer, QdQTransformer
from dmx.compressor.fx import (
    ConfigurationTransformer,
    DMXAwareTransformer,
    RecordInputInterpreter,
)
from dmx.compressor.fx.transformer.utils import dmx_aware_mapping
from torch.fx import GraphModule
from typing import Any, Dict, List, Optional, Union
from inspect import signature


def prepare_tracing_inputs(_model, args, kwargs):
    # remove kwargs with value None
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    # boolean inputs will affect tracing and need to be set as concrete args
    bool_inputs = {k: v for k, v in kwargs.items() if isinstance(v, bool)}
    kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, bool)}

    if hasattr(_model, "old_forward"):
        input_names = (
            signature(_model.old_forward).bind(*args, **kwargs).arguments.keys()
        )
    else:
        input_names = signature(_model.forward).bind(*args, **kwargs).arguments.keys()
    dummy_inputs = {}
    for k in input_names:
        if k not in kwargs:
            dummy_inputs[k] = args[0]
        else:
            dummy_inputs[k] = kwargs[k]
    return input_names, bool_inputs, dummy_inputs


def substitute_transform(
    root: torch.nn.Module,
    concrete_args: Optional[Dict[str, Any]] = None,
    input_names: Optional[List[str]] = None,
    dummy_inputs: Optional[Dict[str, Any]] = None,
    additional_mappings: Optional[Dict[str, Any]] = None,
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
    mod_type = type(root).__module__ + "." + type(root).__name__
    if mod_type in dmx_aware_mapping:
        transformed = dmx_aware_mapping[mod_type].from_raw(root)
        return transformed

    gm, tracer = hf_symbolic_trace(
        root,
        input_names,
        concrete_args=concrete_args,
        dummy_inputs=dummy_inputs,
    )

    gi = RecordInputInterpreter(gm)
    tracing_args, tracing_kwargs = root.tracing_kwargs
    inputs = list(
        prepare_tracing_inputs(root, tracing_args, tracing_kwargs)[2].values()
    )
    gi.run(*inputs)
    transformer = DMXAwareTransformer(
        gm,
        tracer.node_name_to_scope,
        root._gm if root.transformed else None,
        gi.nodeInputs,
    )
    if additional_mappings:
        for target, dmx_module in additional_mappings.items():
            new_target = target.replace("torch.ops.", "")
            transformer.add_dmx_aware_functional_mapping(new_target, dmx_module)
    transformed = transformer.transform()

    return transformed


def qDq_transform(
    root: torch.fx.GraphModule,
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
    # import compiler so q/dq ops are registered

    transformer = QdQTransformer(root)
    transformed = transformer.transform()
    transformed.recompile()
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
