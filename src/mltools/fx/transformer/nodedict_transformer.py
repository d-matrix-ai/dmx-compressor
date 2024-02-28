from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, Set

import torch.fx as fx
from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate
from torch.fx.proxy import Proxy

from .utils import *


class NodeDictTransformer(fx.Transformer):
    """
    A transformer that creates a dict contaning mapping between target of node and the node itself

    Args:
        module (fx.GraphModule): the module to create the mapping

    Attributes:
        module (fx.GraphModule): the module to create the mapping
        nodeDict (dict): dictionary to store the mapping between occurance order of the node and the node
    """

    def __init__(self, module: fx.GraphModule):
        super().__init__(module)
        self.module = module
        # A dictionary to map node.target to node instance
        self.nodeDict = dict()
        self.nodeDict["function"] = list()
        self.nodeDict["method"] = list()
        self.nodeDict["module"] = list()

    def placeholder(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Proxy:
        """
        Execute a ``placeholder`` node. Adds mapping between target of the node and the node itself to self.nodeDict.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            A Proxy containing the node and the tracer of the new graph
        """
        assert isinstance(target, str)
        placeholder_node = self.new_graph.placeholder(target)
        self.nodeDict[placeholder_node.target] = placeholder_node
        return Proxy(placeholder_node, self.tracer)

    def output(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute a ``ouptput`` node. Adds mapping between target of the node and the node itself to self.nodeDict.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            A Proxy containing the node and the tracer of the new graph
        """
        assert isinstance(target, str)
        output_node = self.new_graph.output(target)
        output_node.args = (output_node.prev,)
        self.nodeDict[output_node.target] = output_node
        return Proxy(output_node, self.tracer)

    def get_attr(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Proxy:
        """
        Execute a ``get_attr`` node. Adds mapping between target of the node and the node itself to self.nodeDict.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            A Proxy containing the node and the tracer of the new graph
        """
        assert isinstance(target, str)
        get_attr_node = self.new_graph.get_attr(target)
        self.nodeDict[get_attr_node.target] = get_attr_node
        return Proxy(get_attr_node, self.tracer)

    def call_function(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_function`` node. Adds mapping between target of the node and the node itself to self.nodeDict.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            A Proxy containing the node and the tracer of the new graph
        """
        assert callable(target)
        call_fnc_node = self.new_graph.call_function(target)
        call_fnc_node.args = process_args(args)
        call_fnc_node.kwargs = process_kwargs(kwargs)
        self.nodeDict["function"].append(call_fnc_node)
        return Proxy(call_fnc_node, self.tracer)

    def call_method(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_method`` node. Adds mapping between target of the node and the node itself to self.nodeDict.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            A Proxy containing the node and the tracer of the new graph
        """
        assert isinstance(target, str)
        call_method_node = self.new_graph.call_method(target)
        call_method_node.args = process_args(args)
        call_method_node.kwargs = process_kwargs(kwargs)
        self.nodeDict["method"].append(call_method_node)
        return Proxy(call_method_node, self.tracer)

    def call_module(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_module`` node. Adds mapping between target of the node and the node itself to self.nodeDict.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            A Proxy containing the node and the tracer of the new graph
        """
        assert isinstance(target, str)
        call_module_node = self.new_graph.call_module(target)
        call_module_node.args = process_args(args)
        call_module_node.kwargs = process_kwargs(kwargs)
        self.nodeDict["module"].append(call_module_node)
        return Proxy(call_module_node, self.tracer)

    def transform(self) -> dict:
        """
        Run `module` via interpretation and return the the nodeDict.

        Returns:
            A dictionary containing the mapping between the target of the node and the node itself.
        """
        result = super().run()
        if result is not None:

            def strip_proxy(a: Union[Argument, Proxy]) -> Any:
                return a.node if isinstance(a, Proxy) else a

            self.new_graph.output(map_aggregate(result, strip_proxy))
        return self.nodeDict
