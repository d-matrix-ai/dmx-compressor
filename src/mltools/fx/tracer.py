#!/usr/bin/env python3
#

import torch
from torch import fx
from .. import numerical
from .. import sparse

from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate
from torch.fx.proxy import Proxy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch.fx._compatibility import compatibility
from torch._C import ScriptObject
from types import CodeType, FunctionType, ModuleType
from typing import Any, Dict, NamedTuple, Optional, Set, Tuple, Type, List, Callable, Union
import functools
from torch.fx._symbolic_trace import _autowrap_check,Graph,_Patcher,_patch_wrapped_functions
import transformers.utils.fx as fx_hf

import random
from transformers.models.auto import get_values
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_MAPPING,
    GPT2DoubleHeadsModel,
    PretrainedConfig,
    PreTrainedModel,
    XLNetForQuestionAnswering,
    logging,
)
import inspect

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

def _generate_random_int(low: int = 10, high: int = 20, forbidden_values: Optional[List[int]] = None):
    if forbidden_values is None:
        forbidden_values = []
    value = random.randint(low, high)
    while value in forbidden_values:
        value = random.randint(low, high)
    return value

def _reset_tensor_methods(original_methods: Dict[str, Callable[..., Any]]):
    """Helper function that resets the monkey patched torch.Tensor methods to their original values."""
    for name, method in original_methods.items():
        setattr(torch.Tensor, name, method)

def _function_to_leaf(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrapper that marks func as a leaf function, meaning that it will not be traced through by HFTracer."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def _function_leaf_getter(func_name: str, mapping: Dict[str, Callable[..., Any]]) -> Callable[..., Any]:
    @functools.wraps(mapping[func_name])
    def wrapper(*args, **kwargs):
        return mapping[func_name](*args, **kwargs)

    return wrapper

logger = logging.get_logger(__name__)

class QuantTracer(fx.Tracer):
    """
    Customed tracer with scope manager
    """
    _DEFAULT_METHODS_TO_RECORD = {"__bool__": False, "size": True, "dim": False}
    from transformers import modeling_utils
    _FUNCTIONS_TO_AUTOWRAP = {
        torch: {"arange", "zeros", "ones", "full_like", "eye"},
        modeling_utils.ModuleUtilsMixin: {"create_extended_attention_mask_for_decoder"},
    }

    def __init__(self) -> None:
        self._leaf_functions_register = {}
        for module, names in self._FUNCTIONS_TO_AUTOWRAP.items():
            for name in names:
                self._register_leaf_function(module, name)
        super().__init__()
        self.scope = Scope("model", None)
        self.node_name_to_scope: Dict[str, Tuple[str, type]] = {}
        self.record_stack_traces = True
    
    
    
    def _register_leaf_function(self, module: ModuleType, name: str):
        print("Entered _register_leaf_function")
        """Registers the function called name in module as a leaf function."""
        orig_func = getattr(module, name)
        patched_func = _function_to_leaf(orig_func)
        patched_func.__module__ = __name__
        self._leaf_functions_register[name] = (module, orig_func, patched_func)
    

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
    
    # def _generate_dummy_input(
    #     self, model: PreTrainedModel, input_name: str, shape: List[int]
    # ) -> Dict[str, torch.Tensor]:
    #     print("Entered _generate_dummy_input")
    #     """Generates dummy input for model inference recording."""
    #     model_class = model.__class__
    #     device = model.device
    #     inputs_dict = {}

    #     if input_name in ["labels", "start_positions", "end_positions"]:
    #         batch_size = shape[0]
    #         if model_class in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
    #             inputs_dict["labels"] = torch.zeros(batch_size, dtype=torch.long, device=device)
    #         elif model_class in [
    #             *get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING),
    #             XLNetForQuestionAnswering,
    #         ]:
    #             inputs_dict["start_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
    #             inputs_dict["end_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
    #         elif model_class in [
    #             *get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
    #             *get_values(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING),
    #             *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING),
    #         ]:
    #             inputs_dict["labels"] = torch.zeros(batch_size, dtype=torch.long, device=device)
    #         elif model_class in [
    #             *get_values(MODEL_FOR_PRETRAINING_MAPPING),
    #             *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
    #             *get_values(MODEL_FOR_CAUSAL_LM_MAPPING),
    #             *get_values(MODEL_FOR_MASKED_LM_MAPPING),
    #             *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING),
    #             GPT2DoubleHeadsModel,
    #         ]:
    #             inputs_dict["labels"] = torch.zeros(shape, dtype=torch.long, device=device)
    #         else:
    #             raise NotImplementedError(f"{model_class} not supported yet.")

    #     elif "mask" in input_name or "ids" in input_name:
    #         inputs_dict[input_name] = torch.zeros(shape, dtype=torch.long, device=device)
    #     else:
    #         shape_with_hidden_size = shape + [model.config.hidden_size]
    #         inputs_dict[input_name] = torch.zeros(shape_with_hidden_size, dtype=torch.float, device=device)

    #     return inputs_dict
    
    def trace(self, root: Union[torch.nn.Module, Callable[..., Any]], concrete_args: Optional[Dict[str, Any]] = None,method_names: Optional[Iterable[str]] = None) -> Graph:
        print("Entered trace")
        if concrete_args is None:
            concrete_args = {}

        sig = inspect.signature(root.forward)
        input_names = sig.parameters.keys() - concrete_args.keys()

        # self.record(root, input_names, method_names=method_names)

        # TODO: adapt the way leaf function are wrapped with the "autowrap function" feature from Tracer.
        autowrap_functions = [patched for (_, _, patched) in self._leaf_functions_register.values()]
        self._autowrap_function_ids.update(set([id(f) for f in autowrap_functions]))

        self._patch_leaf_functions_for_root(root)

        self.graph = super().trace(root, concrete_args=concrete_args)

        self._patch_leaf_functions_for_root(root, restore=True)

        # _reset_tensor_methods(self.original_methods)

        # TODO: keep this until necessary.
        # This is necessary because concrete args are added as input to the traced module since
        # https://github.com/pytorch/pytorch/pull/55888.
        # A PR that solves this was posted: https://github.com/pytorch/pytorch/pull/59569 but it was not merged yet.
        # for node in self.graph.nodes:
        #     if node.op == "placeholder":
        #         # Removing default values for inputs as the forward pass will fail with them.
        #         if node.target in input_names:
        #             node.args = ()
        #         # It is a concrete arg so it is not used and should be removed.
        #         else:
        #             self.graph.erase_node(node)

        return self.graph

        # return super().trace(root, concrete_args)
    
    # def record(self, model: PreTrainedModel, input_names: List[str], method_names: Optional[Iterable[str]] = None):
    #     print("Entered record")
    #     """
    #     Records torch.Tensor method outputs (specified by method_names) that will then be used during symbolic tracing.
    #     """
    #     if method_names is None:
    #         method_names = self._DEFAULT_METHODS_TO_RECORD

    #     # Creating a random input shape to generate dummy inputs.
    #     batch_size = _generate_random_int()
    #     sequence_length = _generate_random_int()
    #     shape = [batch_size, sequence_length]

    #     if model.__class__ in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
    #         num_choices = _generate_random_int(low=2, high=5)
    #         shape.insert(1, num_choices)

    #     inputs = {}
    #     for input_name in input_names:
    #         inputs.update(self._generate_dummy_input(model, input_name, shape))
        
    #     cache_names, original_methods = self._monkey_patch_tensor_methods_for_model_recording(model, method_names)
    #     self.original_methods = original_methods

    #     model(**inputs)

    #     _reset_tensor_methods(original_methods)

    #     self.recorded_methods = {
    #         method_name: cache_name for method_name, cache_name in cache_names.items() if hasattr(model, cache_name)
    #     }

    def _patch_leaf_functions_for_root(self, root: PreTrainedModel, restore: bool = False):
        print("Entered _patch_leaf_functions_for_root")
        """Patches leaf functions specifically for root."""
        for name in self._leaf_functions_register:
            module, orig_func, patched_func = self._leaf_functions_register[name]
            if restore:
                root.__class__.forward.__globals__.pop(name)
                setattr(module, name, orig_func)
            else:
                root.__class__.forward.__globals__[name] = patched_func
                leaf_getter = _function_leaf_getter(name, root.__class__.forward.__globals__)
                leaf_getter.__module__ = __name__
                setattr(module, name, leaf_getter)

    # def _monkey_patch_tensor_methods_for_model_recording(self, model: PreTrainedModel, method_names: Iterable[str]):
    #     print("Entered _monkey_patch_tensor_methods_for_model_recording")
    #     """
    #     Helper function that patches torch.Tensor methods (specified by the method_names list) to record model
    #     inference before symbolic tracing.
    #     """
    #     cache_names = {}
    #     original_methods = {}
    #     module_ids = set(id(mod) for mod in model.modules())
    #     for method_name in method_names:
    #         cache_name = f"cache_{method_name}"
    #         cache_names[method_name] = cache_name
    #         if not hasattr(torch.Tensor, method_name):
    #             logger.info(f"torch.Tensor has no method called {method_name}, skipping patching.")
    #             continue
    #         original_methods[method_name] = getattr(torch.Tensor, method_name)
    #         setattr(
    #             torch.Tensor,
    #             method_name,
    #             self._wrap_method_for_model_recording(model, method_name, cache_name, module_ids),
    #         )

    #         if method_name == "size":
    #             original_methods["shape"] = torch.Tensor.shape
    #             setattr(torch.Tensor, "shape", property(getattr(torch.Tensor, method_name)))

    #     return cache_names, original_methods 
    
    # def _wrap_method_for_model_recording(
    #     self, model: PreTrainedModel, method_name: str, cache_name: str, module_ids: List[int]
    # ):
    #     print("Entered _wrap_method_for_model_recording")
    #     """Helper function that wraps a torch.Tensor method to record its outputs during forward pass."""
    #     method = getattr(torch.Tensor, method_name)

    #     @functools.wraps(method)
    #     def wrapped(*args, **kwargs):
    #         if self._method_is_called_in_leaf_module(module_ids):
    #             return method(*args, **kwargs)
    #         if not hasattr(model, cache_name):
    #             setattr(model, cache_name, [])
    #         cache = getattr(model, cache_name)
    #         res = method(*args, **kwargs)
    #         cache.append(res)
    #         return res

    #     return wrapped
    
    # def _method_is_called_in_leaf_module(self, module_ids: List[int]) -> bool:
    #     print("Entered _method_is_called_in_leaf_module")
    #     """
    #     Finds out if the method (that is being recorded) is called inside a leaf module, this allows to not record
    #     outputs that will not be encountered by the tracer.
    #     """

    #     currentframe = inspect.currentframe()
    #     while currentframe:
    #         if currentframe is None:
    #             return False
    #         module = currentframe.f_locals.get("self", None)
    #         if id(module) in module_ids and self.is_leaf_module(module, "Not used anyway"):
    #             return True
    #         currentframe = currentframe.f_back
    #     return False
    
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
            logger = logging.get_logger(__name__)
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
    