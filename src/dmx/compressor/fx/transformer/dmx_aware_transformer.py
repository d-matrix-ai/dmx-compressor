from typing import Any, Dict, Tuple
import torch.fx as fx
from torch.fx.node import Argument, Target
from torch.fx.proxy import Proxy

import inspect
from .utils import *


class DMXAwareTransformer(fx.Transformer):
    """
    Substitute as in dmx.model.aware(), replace torch.nn.modules and
    activations with dmx counterpart

    Args:
        module (fx.GraphModule): the module to transform
        node_name_to_scope (dict): A dictionary storing the mapping between node names and scopes

    Attributes:
        module (fx.GraphModule): the module to transform
        node_name_to_scope (dict): A dictionary storing the mapping between node names and scopes
        old_gm (fx.GraphModule): dmxmodules to reuse if module was already transformed
    """

    def __init__(
        self,
        module: fx.GraphModule,
        node_name_to_scope: dict,
        old_gm: fx.GraphModule = None,
    ):
        super().__init__(module)
        self.module = module
        self.node_name_to_scope = node_name_to_scope
        self.old_gm = old_gm

    def call_module(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_module`` node, replaces the module with its dmx counterpart and returns the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            A Proxy containing the new node and the tracer of the new graph

        """
        assert isinstance(target, str)
        curr_mod = self.module.get_submodule(target)
        node_key = type(curr_mod).__module__ + "." + type(curr_mod).__name__
        if node_key not in dmx_aware_mapping:
            return super().call_module(target, args, kwargs)
        self.add_submod(target, dmx_aware_mapping[node_key].from_raw(curr_mod))
        new_node = self.new_graph.create_node(
            "call_module", target, args=(args[0].node,)
        )
        return Proxy(new_node, self.tracer)

    def add_submod(self, name, mod):
        """
        this function will try to reuse modules in old_gm if possible
        """
        try:
            self.module.add_submodule(name, self.old_gm.get_submodule(name))
        except:
            self.module.add_submodule(name, mod)

    def call_method(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if target == "baddbmm":
            candidate = target
            curr_name = get_name_for_func_nodes(
                target,
                self.new_graph._graph_namespace._used_names,
                self.new_graph._graph_namespace._base_count,
            )
            scope, _ = self.node_name_to_scope[curr_name]
            new_name = scope + "." + candidate if scope != "" else candidate

            # If new name is not candidate, need to add candidate to used names,
            # otherwise next call_method will use the same candidate. (create_name is also called in create_node)
            if new_name != candidate:
                self.new_graph._graph_namespace.create_name(candidate, None)
            from dmx.compressor.modeling.nn import BAddBMM

            self.add_submod(new_name, BAddBMM())
            new_node = self.new_graph.create_node(
                "call_module",
                new_name,
            )
            new_node.args = process_args(args)
            new_node.kwargs = process_kwargs(kwargs)
            return Proxy(new_node, self.tracer)
        else:
            return super().call_method(target, args, kwargs)

    def create_unique_name_in_scope(self, cand_name):
        # output of get_name_for_func_nodes will replace all . with _
        curr_name = get_name_for_func_nodes(
            cand_name,
            self.new_graph._graph_namespace._used_names,
            self.new_graph._graph_namespace._base_count,
        )
        new_name = (
            cand_name + "_" + curr_name[curr_name.rfind("_") + 1 :]
            if curr_name[-1].isdigit()
            else cand_name
        )
        return new_name

    def call_function(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_function`` node, replaces the function with its dmx counterpart and returns the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            A Proxy containing the new node and the tracer of the new graph
        """
        assert callable(target)
        node_key = str(target)
        if node_key not in dmx_aware_functional_mappings:
            return super().call_function(target, args, kwargs)

        candidate = self.new_graph._target_to_str(target)
        curr_name = get_name_for_func_nodes(
            candidate,
            self.new_graph._graph_namespace._used_names,
            self.new_graph._graph_namespace._base_count,
        )
        curr_target, curr_type = self.node_name_to_scope[curr_name]
        if node_key == "<built-in function add>":
            if (
                isinstance(args[0], Proxy)
                and isinstance(args[1], Proxy)
                and args[0].node.op
                in ["call_module", "call_function", "call_method", "placeholder"]
                and args[1].node.op
                in ["call_module", "call_function", "call_method", "placeholder"]
            ):
                cand_name = curr_target + ".resadd" if curr_target != "" else "resadd"
            else:
                return super().call_function(target, args, kwargs)
        elif node_key in "<built-in function mul>":
            if (
                isinstance(args[0], Proxy)
                and isinstance(args[1], Proxy)
                and args[0].node.op
                in ["call_module", "call_function", "call_method", "placeholder"]
                and args[1].node.op
                in ["call_module", "call_function", "call_method", "placeholder"]
            ):
                cand_name = curr_target + ".mul" if curr_target != "" else curr_name
            else:
                return super().call_function(target, args, kwargs)
        elif node_key in [
            repr(eval("torch.matmul")),
            repr(eval("torch.bmm")),
            "<built-in function matmul>",
        ]:
            cand_name = curr_target + ".actmatmul" if curr_target != "" else "actmatmul"
        else:
            cand_name = (
                curr_target + "." + candidate if curr_target != "" else curr_name
            )
        # when cand_name is not the same as curr_name, it did not go through the create_unique_name_in_scope process, so we need to do it here.
        # We also need to add curr_name to used names, otherwise next call_function will use the same curr_name, which will map to the wrong node in the old graph. (create_name is also called in create_node
        if cand_name != curr_name:
            new_name = self.create_unique_name_in_scope(cand_name)
            self.new_graph._graph_namespace.create_name(curr_name, None)
        else:
            new_name = curr_name
        # find out what kwargs to pass in to new module init, which kwargs to pass into forward function of module
        empty_mod = dmx_aware_functional_mappings[node_key]()
        accepted_kwarg_keys = inspect.signature(empty_mod.__init__).parameters.keys()
        initkwargs = {}
        newkwargs = {}
        for key, value in kwargs.items():
            if key in accepted_kwarg_keys:
                initkwargs[key] = value
            else:
                newkwargs[key] = value
        self.add_submod(new_name, dmx_aware_functional_mappings[node_key](**initkwargs))
        new_node = self.new_graph.create_node(
            "call_module",
            new_name,
        )
        new_node.args = process_args(args)
        new_node.kwargs = process_kwargs(newkwargs)
        return Proxy(new_node, self.tracer)
