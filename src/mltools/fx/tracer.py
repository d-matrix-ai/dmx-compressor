#!/usr/bin/env python3
#
import logging
import torch
from torch import fx
from .. import numerical
from .. import sparse

from torch.fx.node import Argument, Node, Target
from typing import Any, Callable, Dict, Optional, Tuple
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Callable,
)

import transformers.utils.fx as fx_hf


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
