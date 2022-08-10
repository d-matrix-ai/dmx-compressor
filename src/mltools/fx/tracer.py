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
from typing import Any, Dict, NamedTuple, Optional, Set, Tuple, Type, List, Callable, Union
import functools
from torch.fx._symbolic_trace import _autowrap_check,Graph,_CPatchManager,_Patcher,_patch_wrapped_functions

_orig_module_call : Callable = torch.nn.Module.__call__
_orig_module_getattr : Callable = torch.nn.Module.__getattr__

# Referenced from https://pytorch.org/docs/stable/_modules/torch/ao/quantization/quantize_fx.html#convert_fx
class Scope(object):
    """ Scope object that records the module path and the module type
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
    """ A context manager to track the Scope of Node during symbolic tracing.
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
    Customed tracer with scope manager
    """
    def __init__(self) -> None:
        super().__init__()
        self.scope = Scope("", None)
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
        return (
            (
                is_leaf
            )
        )
    
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
            logger.info("path:",self.scope.module_path)
            logger.info("type:",self.scope.module_type)
            return super().call_module(m,forward,args,kwargs)

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
    
    def create_arg(self, a: Any) -> 'Argument':
        return super().create_arg(a)
    
    def create_args_for_root(self, root_fn, is_module, concrete_args=None):
        return super().create_args_for_root(root_fn, is_module, concrete_args)

    @compatibility(is_backward_compatible=True)
    def trace(self, root: Union[torch.nn.Module, Callable[..., Any]], concrete_args: Optional[Dict[str, Any]] = None) -> Graph:
        """
        Trace ``root`` and return the corresponding FX ``Graph`` representation. ``root``
        can either be an ``nn.Module`` instance or a Python callable.

        Note that after this call, ``self.root`` may be different from the ``root`` passed
        in here. For example, when a free function is passed to ``trace()``, we will
        create an ``nn.Module`` instance to use as the root and add embedded constants
        to.


        Args:

            root (Union[Module, Callable]): Either a ``Module`` or a function to be
                traced through. Backwards-compatibility for this parameter is
                guaranteed.
            concrete_args (Optional[Dict[str, any]]): Concrete arguments that should
                not be treated as Proxies. This parameter is experimental and
                its backwards-compatibility is *NOT* guaranteed.

        Returns:

            A ``Graph`` representing the semantics of the passed-in ``root``.
        """
        if isinstance(root, torch.nn.Module):
            self.root = root
            fn = type(root).forward
            self.submodule_paths = {mod: name for name, mod in root.named_modules()}
        else:
            self.root = torch.nn.Module()
            fn = root

        tracer_cls: Optional[Type['fx.Tracer']] = getattr(self, '__class__', None)
        self.graph = Graph(tracer_cls=tracer_cls)

        # When we encounter a Tensor value that's not a parameter, we look if it
        # is some other attribute on the model. Construct a dict mapping Tensor
        # values to the qualified name here for efficiency. This is used downstream
        # in create_arg
        self.tensor_attrs : Dict[Union[torch.Tensor, ScriptObject], str] = {}

        def collect_tensor_attrs(m : torch.nn.Module, prefix_atoms : List[str]):
            for k, v in m.__dict__.items():
                if isinstance(v, (torch.Tensor, ScriptObject)):
                    self.tensor_attrs[v] = '.'.join(prefix_atoms + [k])
            for k, v in m.named_children():
                collect_tensor_attrs(v, prefix_atoms + [k])

        collect_tensor_attrs(self.root, [])
        assert isinstance(fn, FunctionType)

        fn_globals = fn.__globals__  # run before it gets patched
        fn, args = self.create_args_for_root(fn, isinstance(root, torch.nn.Module), concrete_args)

        parameter_proxy_cache : Dict[str, Proxy] = {}  # Reduce number of get_attr calls

        # Method dispatch on parameters is not recorded unless it's directly used.
        # Thus, we need to insert a proxy when __getattr__ requests a parameter.
        @functools.wraps(_orig_module_getattr)
        def module_getattr_wrapper(mod, attr):
            attr_val = _orig_module_getattr(mod, attr)
            return self._module_getattr(attr, attr_val, parameter_proxy_cache)

        @functools.wraps(_orig_module_call)
        def module_call_wrapper(mod, *args, **kwargs):
            def forward(*args, **kwargs):
                return _orig_module_call(mod, *args, **kwargs)

            _autowrap_check(patcher, getattr(getattr(mod, "forward", mod), "__globals__", {}),
                            self._autowrap_function_ids)
            return self.call_module(mod, forward, args, kwargs)

        with _CPatchManager(self):
            with _Patcher() as patcher:
                # allow duplicate patches to support the case of nested calls
                patcher.patch_method(torch.nn.Module, "__getattr__", module_getattr_wrapper, deduplicate=False)
                patcher.patch_method(torch.nn.Module, "__call__", module_call_wrapper, deduplicate=False)
                _patch_wrapped_functions(patcher)
                _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
                for module in self._autowrap_search:
                    _autowrap_check(patcher, module.__dict__, self._autowrap_function_ids)
                self.create_node('output', 'output', (self.create_arg(fn(*args)),), {},
                                 type_expr=fn.__annotations__.get('return', None))

        self.submodule_paths = None

        return self.graph