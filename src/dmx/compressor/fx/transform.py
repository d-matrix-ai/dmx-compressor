import torch
import torch.nn as nn

from dmx.compressor.fx.tracer import HFQuantTracer, symbolic_trace, hf_symbolic_trace
from ..fx import QuantTracer, InputOutputTransformer, QdQTransformer
from dmx.compressor.fx import ConfigurationTransformer, DMXAwareTransformer
from dmx.compressor.fx.transformer.utils import dmx_aware_mapping
from torch.fx import GraphModule
from typing import Any, Dict, List, Optional, Union
from transformers.modeling_utils import get_parameter_device
import inspect


def substitute_transform(
    root: torch.nn.Module,
    concrete_args: Optional[Dict[str, Any]] = None,
    hf: bool = False,
    input_names: Optional[List[str]] = None,
    dummy_inputs: Optional[Dict[str, Any]] = None,
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
    root = remove_new_forward(root)
    if not hasattr(root, "config"):
        root.config = None
    if not hf:
        root.device = get_parameter_device(root)
        # set dummy_input for params without default
        sig = inspect.signature(
            root.forward if isinstance(root, torch.nn.Module) else root
        )
        for param in sig.parameters.values():
            if param.name in dummy_inputs:
                continue
            if param.default is inspect.Parameter.empty:
                dummy_inputs[param.name] = None
        gm, tracer = hf_symbolic_trace(
            root, input_names, concrete_args=concrete_args, dummy_inputs=dummy_inputs
        )
    else:
        gm, tracer = hf_symbolic_trace(root, input_names, concrete_args=concrete_args)
    transformer = DMXAwareTransformer(
        gm, tracer.node_name_to_scope, root._gm if root.transformed else None
    )
    transformed = transformer.transform()
    # Copy over all object attributes (i.e. config files)
    for key, val in root.__dict__.items():
        if key not in transformed.__dict__ and key != "forward":
            transformed.__dict__[key] = val
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
    import dmir_compiler.custom_ops

    transformer = QdQTransformer(root)
    transformed = transformer.transform()
    transformed.recompile()
    return transformed


def remove_new_forward(model):
    """
    This function removes the .forward attributes assigned to submodules of model which is added for model parallelization.
    """
    for m in model.modules():
        if hasattr(m, "_old_forward"):
            m.forward = m._old_forward
    if hasattr(model, "_old_forward"):
        model.forward = model._old_forward
    return model


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
