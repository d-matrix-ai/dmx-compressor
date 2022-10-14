#!/usr/bin/env python3
#
import logging
import torch
from torch import fx
from .. import numerical
from .. import sparse

from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate
from torch.fx.proxy import Proxy
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from torch.fx._compatibility import compatibility
from torch._C import ScriptObject
from types import CodeType, FunctionType, ModuleType
from typing import (
    Any,
    Dict,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    List,
    Callable,
    Union,
)
import functools
from torch.fx._symbolic_trace import (
    _autowrap_check,
    Graph,
    _Patcher,
    _patch_wrapped_functions,
)

import transformers.utils.fx as fx_hf

_orig_module_call: Callable = torch.nn.Module.__call__
_orig_module_getattr: Callable = torch.nn.Module.__getattr__

# Referenced from https://pytorch.org/docs/stable/_modules/torch/ao/quantization/quantize_fx.html#convert_fx
class Scope(object):
    """Scope object that records the module path and the module type
    of a module. Scope is used to track the information of the module
    that contains a Node in a Graph of GraphModule. For example::

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

    """

    def __init__(self, module_path: str, module_type: Any):
        super().__init__()
        self.module_path = module_path
        self.module_type = module_type


class ScopeContextManager(object):
    """A context manager to track the Scope of Node during symbolic tracing.
    When entering a forward function of a Module, we'll update the scope information of
    the current module, and when we exit, we'll restore the previous scope information.
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
        self.scope.module_path = self.prev_module_path
        self.scope.module_type = self.prev_module_type
        return


class QuantTracer(fx.Tracer):
    """
    Customed tracer with scope manager and returns a flat GraphModule
    """

    def __init__(self) -> None:
        super().__init__()
        self.scope = Scope("model", None)
        self.node_name_to_scope: Dict[str, Tuple[str, type]] = {}
        self.record_stack_traces = True

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:

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
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        self.node_name_to_scope[node.name] = (
            self.scope.module_path,
            self.scope.module_type,
        )
        return node


class HFQuantTracer(fx_hf.HFTracer):
    """
    Customed tracer with scope manager for HuggingFace
    """

    def __init__(self) -> None:
        super().__init__()
        self.scope = Scope("model", None)
        self.node_name_to_scope: Dict[str, Tuple[str, type]] = {}
        self.record_stack_traces = True

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:

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
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        self.node_name_to_scope[node.name] = (
            self.scope.module_path,
            self.scope.module_type,
        )
        return node
