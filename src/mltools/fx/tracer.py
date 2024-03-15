#!/usr/bin/env python3
#
import logging
import math
import torch
from torch import fx
from .. import numerical
from .. import sparse
import warnings

from torch.fx.node import Argument, Node, Target
from typing import Any, Callable, Dict, Optional, Tuple, Type, List, Union
import transformers.utils.fx as fx_hf
import transformers

from transformers.utils.fx import (
    HFTracer,
    get_concrete_args,
    is_model_supported,
    _SUPPORTED_MODELS,
)
from torch.fx.graph_module import GraphModule
from transformers.modeling_utils import PreTrainedModel


class DmxHFTracer(HFTracer):
    """
    Custom HFTracer where definition of leaf nodes
    """

    def __init__(self, autowrap_modules=(math,), autowrap_functions=()):
        super().__init__(
            autowrap_modules=autowrap_modules, autowrap_functions=autowrap_functions
        )

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.

        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.

        Args:
            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        Returns:
            True if m is a leaf module
        """
        is_leaf = isinstance(
            m,
            (
                numerical.CastTo,
                sparse.Sparsify,
                transformers.pytorch_utils.Conv1D,
            ),
        )
        is_leaf = is_leaf or (
            (
                m.__module__.startswith("torch.nn")
                or m.__module__.startswith("torch.ao.nn")
                or m.__module__.startswith("transformers.activations")
            )
            and not isinstance(m, torch.nn.Sequential)
        )
        return is_leaf


def hf_symbolic_trace(
    model: PreTrainedModel,
    input_names: Optional[List[str]] = None,
    concrete_args: Optional[Dict[str, Any]] = None,
    tracer_cls: Type[DmxHFTracer] = DmxHFTracer,
) -> GraphModule:
    """
    Performs symbolic tracing on a huggingface model.

    Args:
        model ([`PretrainedModel`]):
            The model to trace.
        input_names (`List[str]`, *optional*):
            The names of the inputs of the traced model. If unset, model.dummy_inputs.keys() are used instead.
        disable_check (`bool`, *optional*, defaults to `False`):
            If `True`, no check is done before trying to trace the model, this is mostly usesul for debugging purposes.
        tracer_cls (`Type[HFTracer]`, *optional*, defaults to `HFTracer`):
            The tracer class to use for instantiating the tracer. If unset, `HFTracer` is used instead.

    Returns:
        `torch.fx.GraphModule`: A GraphModule constructed by recording operations seen while tracing the model.
        `torch.fx.Tracer`: The tracer used for tracing the model.

    Example:

        ```python
        from mltools.fx.tracer import hf_symbolic_trace

        traced_model,tracer = hf_symbolic_trace(model, input_names=["input_ids", "attention_mask", "token_type_ids"])
        ```
    """
    if input_names is None:
        input_names = model.dummy_inputs.keys()

    input_names = list(input_names)
    if concrete_args:
        new_args = get_concrete_args(model, input_names)
        for key, value in new_args.items():
            if key not in concrete_args:
                concrete_args[key] = value
        concrete_args.update()
    else:
        concrete_args = get_concrete_args(model, input_names)

    # Tracing.
    tracer = tracer_cls()
    traced_graph = tracer.trace(model, concrete_args=concrete_args)
    traced = GraphModule(model, traced_graph)

    traced.config = model.config
    # The model class must be stored as an attribute to allow model deserialization, which uses trace, and thus
    # _generate_dummy_input, where the model class is needed.
    traced.class_for_deserialization = model.__class__
    traced.device = model.device

    return (traced, tracer)


def symbolic_trace(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[Dict[str, Any]] = None,
) -> GraphModule:
    """
    Symbolic tracing API

    Given an ``nn.Module`` or function instance ``root``, this function will return a ``GraphModule``
    constructed by recording operations seen while tracing through ``root`` and the tracer used to trace the model.

    ``concrete_args`` allows you to partially specialize your function, whether it's to remove control flow or data structures.

    For example::

        def f(a, b):
            if b == True:
                return a
            else:
                return a*2

    FX can typically not trace through this due to the presence of control
    flow. However, we can use `concrete_args` to specialize on the value of
    `b` to trace through this::

        f = fx.symbolic_trace(f, concrete_args={'b': False})
        assert f(3, False)  == 6

    Note that although you can still pass in different values of `b`, they will be ignored.

    We can also use `concrete_args` to eliminate data-structure handling from
    our function. This will use pytrees to flatten your input. To avoid
    overspecializing, pass in `fx.PH` for values that shouldn't be
    specialized. For example::

        def f(x):
            out = 0
            for v in x.values():
                out += v
            return out
        f = fx.symbolic_trace(f, concrete_args={'x': {'a': fx.PH, 'b': fx.PH, 'c': fx.PH}})
        assert f({'a': 1, 'b': 2, 'c': 4}) == 7


    Args:
        root (Union[torch.nn.Module, Callable]): Module or function to be traced and converted
            into a Graph representation.
        concrete_args (Optional[Dict[str, any]]): Inputs to be partially specialized

    Returns:
        GraphModule: a Module created from the recorded operations from ``root``.
        Tracer: the tracer used for tracing the model
    """
    tracer = fx.Tracer()
    graph = tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    return (GraphModule(tracer.root, graph, name), tracer)


# Referenced from https://pytorch.org/docs/stable/_modules/torch/ao/quantization/quantize_fx.html#convert_fx
class Scope(object):
    """Scope object that records the module path and the module type
    of a module. Scope is used to track the information of the module
    that contains a Node in a Graph of GraphModule.

    Example:
        class Sub(torch.nn.Module):
            def forward(self, x):
                # This will be a call_method Node in GraphModule,
                # scope for this would be (module_path="sub", module_type=Sub)
                return x.transpose(1, 2)

        class M(torch.nn.Module):
            def __init__(self):
                self.sub = Sub()

            def forward(self, x):
                # This will be a call_method Node as well,
                # scope for this would be (module_path="", None)
                x = x.transpose(1, 2)
                x = self.sub(x)
                return x

    Args:
        module_path (str): String describing the path to the module
        module_type (Any): type of the module

    Attributes:
        module_path (str): String describing the path to the module
        module_type (Any): type of the module


    """

    def __init__(self, module_path: str, module_type: Any):
        super().__init__()
        self.module_path = module_path
        self.module_type = module_type


class ScopeContextManager(object):
    """A context manager to track the Scope of Node during symbolic tracing.
    When entering a forward function of a Module, we'll update the scope information of
    the current module, and when we exit, we'll restore the previous scope information.

    Args:
        scope (Scope): Scope object to store the module details
        current_module (torch.nn.Module): Current module object
        current_module_path (str): String path to current module

    Attributes:
        prev_module_type (Any): Type of the previous module
        prev_module_path (str): String path to previous module
        scope (Scope): Scope object to store the module details
    """

    def __init__(
        self, scope: Scope, current_module: torch.nn.Module, current_module_path: str
    ):
        super().__init__()
        self.prev_module_type = scope.module_type
        self.prev_module_path = scope.module_path
        self.scope = scope
        self.scope.module_path = self.prev_module_path + "__" + current_module_path
        self.scope.module_type = type(current_module)

    def __enter__(self):
        return

    def __exit__(self, *args):
        """Restore information of the previous module on exit."""
        self.scope.module_path = self.prev_module_path
        self.scope.module_type = self.prev_module_type
        return


class QuantTracer(fx.Tracer):
    """
    Customed tracer with scope manager and returns a flat GraphModule

    Attributes:
        scope (Scope): Scope object to record the path and type of a module
        node_name_to_scope (Dict): Dictionary that maps node name to scope
        record_stack_traces (bool): Not in use yet
    """

    def __init__(self) -> None:
        super().__init__()
        self.scope = Scope("model", None)
        self.node_name_to_scope: Dict[str, Tuple[str, type]] = {}
        self.record_stack_traces = True

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.

        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.

        Args:
            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.

        Returns:
            True if m is a leaf module
        """

        is_leaf = isinstance(
            m,
            (
                numerical.CastTo,
                sparse.Sparsify,
            ),
        )
        return is_leaf

    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """
        Method that specifies the behavior of this ``Tracer`` when it encounters
        a call to an ``nn.Module`` instance.

        By default, the behavior is to check if the called module is a leaf module
        via ``is_leaf_module``. If it is, emit a ``call_module`` node referring to
        ``m`` in the ``Graph``. Otherwise, call the ``Module`` normally, tracing through
        the operations in its ``forward`` function.

        Args:
            m (Module): The module for which a call is being emitted
            forward (Callable): The forward() method of the ``Module`` to be invoked
            args (Tuple): args of the module callsite
            kwargs (Dict): kwargs of the module callsite

        Returns:
            The return value from the Module call. In the case that a ``call_module``
            node was emitted, this is a ``Proxy`` value. Otherwise, it is whatever
            value was returned from the ``Module`` invocation.
        """
        module_qualified_name = self.path_of_module(m)
        # Creating scope with information of current module
        # scope will be restored automatically upon exit
        with ScopeContextManager(self.scope, m, module_qualified_name):
            logger = logging.getLogger(__name__)
            logger.info("path:", self.scope.module_path)
            logger.info("type:", self.scope.module_type)
            return super().call_module(m, forward, args, kwargs)

    def create_node(
        self,
        kind: str,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> Node:
        """
        Inserts a graph node given target, args, kwargs, and name.

        Args:
            op (str): the opcode for this Node. One of 'call_function', 'call_method', 'get_attr',
                'call_module', 'placeholder', or 'output'. The semantics of these opcodes are
                described in the ``Graph`` docstring.
            args (Optional[Tuple[Argument, ...]]): is a tuple of arguments to this node.
            kwargs (Optional[Dict[str, Argument]]): the kwargs of this Node
            name (Optional[str]): an optional string name for the ``Node``.
                This will influence the name of the value assigned to in the
                Python generated code.
            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:
            The newly-created and inserted node.
        """
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        self.node_name_to_scope[node.name] = (
            self.scope.module_path,
            self.scope.module_type,
        )
        return node


class HFQuantTracer(fx_hf.HFTracer):
    """
    Customed tracer with scope manager for HuggingFace

    Attributes:
        scope (Scope): Scope object to record the path and type of a module
        node_name_to_scope (Dict): Dictionary that maps node name to scope
        record_stack_traces (bool): Not in use yet
    """

    def __init__(self) -> None:
        super().__init__()
        self.scope = Scope("model", None)
        self.node_name_to_scope: Dict[str, Tuple[str, type]] = {}
        self.record_stack_traces = True

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.

        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.

        Args:
            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        Returns:
            True if m is a leaf module
        """
        is_leaf = isinstance(
            m,
            (
                numerical.CastTo,
                sparse.Sparsify,
            ),
        )
        return is_leaf

    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """
        Method that specifies the behavior of this ``Tracer`` when it encounters
        a call to an ``nn.Module`` instance.

        By default, the behavior is to check if the called module is a leaf module
        via ``is_leaf_module``. If it is, emit a ``call_module`` node referring to
        ``m`` in the ``Graph``. Otherwise, call the ``Module`` normally, tracing through
        the operations in its ``forward`` function.

        Args:
            m (Module): The module for which a call is being emitted
            forward (Callable): The forward() method of the ``Module`` to be invoked
            args (Tuple): args of the module callsite
            kwargs (Dict): kwargs of the module callsite

        Returns:
            The return value from the Module call. In the case that a ``call_module``
            node was emitted, this is a ``Proxy`` value. Otherwise, it is whatever
            value was returned from the ``Module`` invocation.
        """
        module_qualified_name = self.path_of_module(m)
        # Creating scope with information of current module
        # scope will be restored automatically upon exit
        with ScopeContextManager(self.scope, m, module_qualified_name):
            logger = logging.getLogger(__name__)
            logger.info("path:", self.scope.module_path)
            logger.info("type:", self.scope.module_type)
            return super().call_module(m, forward, args, kwargs)

    def create_node(
        self,
        kind: str,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> Node:
        """
        Inserts a graph node given target, args, kwargs, and name.

        Args:
            op (str): the opcode for this Node. One of 'call_function', 'call_method', 'get_attr',
                'call_module', 'placeholder', or 'output'. The semantics of these opcodes are
                described in the ``Graph`` docstring.
            args (Optional[Tuple[Argument, ...]]): is a tuple of arguments to this node.
            kwargs (Optional[Dict[str, Argument]]): the kwargs of this Node
            name (Optional[str]): an optional string name for the ``Node``.
                This will influence the name of the value assigned to in the
                Python generated code.
            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:
            The newly-created and inserted node.
        """
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        self.node_name_to_scope[node.name] = (
            self.scope.module_path,
            self.scope.module_type,
        )
        return node
