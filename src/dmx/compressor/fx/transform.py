import torch
import torch.nn as nn

from dmx.compressor.fx.tracer import HFQuantTracer, hf_symbolic_trace
from ..fx import (
    QuantTracer,
    InputOutputTransformer,
    QdQTransformer,
    ExportSubstituteTransformer,
)
from dmx.compressor.fx import (
    ConfigurationTransformer,
    DMXAwareTransformer,
    RecordInputInterpreter,
    RecordInputInterpreterExport,
)
from dmx.compressor.fx.transformer.utils import *
from torch.fx import GraphModule
from typing import Any, Dict, List, Optional, Union
from inspect import signature
import warnings
from collections.abc import Iterable
import dmx.compressor.modeling.nn as dmxnn


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
    i = 0
    for k in input_names:
        if k not in kwargs:
            dummy_inputs[k] = args[i]
            i += 1
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
    gi.run(*list(dummy_inputs.values()))
    transformer = DMXAwareTransformer(
        gm,
        tracer.node_name_to_scope,
        (
            list(root._gms.values())
            if root.transformed and hasattr(root, "_gms")
            else None
        ),
        gi.nodeInputs,
    )
    if additional_mappings:
        for target, dmx_module in additional_mappings.items():
            new_target = target.replace("torch.ops.", "")
            transformer.add_dmx_aware_functional_mapping(new_target, dmx_module)
    transformed = transformer.transform()

    return transformed


def prepare_module_call_signature(model: torch.nn.Module) -> tuple:
    # Prepare the preserve_module_call_signature argument for export, this is needed for modules with more than 1 input
    result = []
    for n, m in model.named_modules():
        mod_type = type(m).__module__ + "." + type(m).__name__
        if mod_type in dmx_aware_mapping and issubclass(
            dmx_aware_mapping[mod_type], dmxnn.RotaryEmbedding
        ):
            result.append(n)
    return tuple(result)


def prepare_dynamic_shapes(model, kwargs) -> dict:
    def create_dynamic_shape(arguments):
        if isinstance(arguments, torch.Tensor):
            return {i: torch.export.Dim.AUTO for i in range(arguments.dim())}
        elif isinstance(arguments, dict):
            return {k: create_dynamic_shape(v) for k, v in arguments.items()}
        elif isinstance(arguments, Iterable):
            return type(arguments)([create_dynamic_shape(v) for v in arguments])
        else:
            return None

    binded_kwargs = signature(model.forward).bind(**kwargs).arguments
    return create_dynamic_shape(binded_kwargs)


stric_export_class = (
    transformers.models.whisper.modeling_whisper.WhisperPreTrainedModel,
)


def export_substitute_transform(
    root: torch.nn.Module, kwargs, additional_mappings=None
):
    print("Export triggered")
    mod_type = type(root).__module__ + "." + type(root).__name__
    if mod_type in dmx_aware_mapping:
        transformed = dmx_aware_mapping[mod_type].from_raw(root)
        return transformed
    if hasattr(root, "hf_device_map") and len(root.hf_device_map) > 1:

        raise ValueError(
            "Export does not work with models dispatched on multiple devices! Please set device_map='cuda'"
        )

    call_sig = prepare_module_call_signature(root)
    dynamic_shapes = prepare_dynamic_shapes(root, kwargs)
    # An artifact of assigning forward function, messes up with export for CLIP
    delattr(root, "forward")
    export_module = torch.export.export(
        root,
        (),
        kwargs,
        strict=isinstance(root, stric_export_class),
        preserve_module_call_signature=call_sig,
        dynamic_shapes=dynamic_shapes,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gi = RecordInputInterpreterExport(export_module.module())
        gi.run(*list(kwargs.values()))
        unflattened = torch.export.unflatten(export_module)
    unflattened.check_input_constraints = False

    transformer = ExportSubstituteTransformer(
        torch.fx.GraphModule(unflattened, unflattened.graph),
        root,
        node_inputs=gi.nodeInputs,
    )
    if additional_mappings:
        for target, dmx_module in additional_mappings.items():
            new_target = target.replace("torch.ops.", "")
            transformer.dmx_aware_function_mapping_export[new_target] = dmx_module
    transformed = transformer.transform()
    transformed.graph.eliminate_dead_code()
    transformed.out_sig = export_module.module_call_graph[0].signature.out_spec
    return transformed


def qDq_transform(
    root: torch.fx.GraphModule,
):
    """
    A function that transforms the model by substituting CastTos with Q/dQ ops

    Args:
        root (torch.nn.Module): model/module to transform

    Returns:
        transformed model
    """
    # import compiler so q/dq ops are registered

    transformer = QdQTransformer(root)
    transformed = transformer.transform()
    transformed.recompile()
    return transformed


make_compiler_graph = qDq_transform


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
