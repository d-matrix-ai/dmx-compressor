from typing import Any, Dict, Tuple

import torch.fx as fx
from torch.fx.node import Argument, Target
from torch.fx.proxy import Proxy

from dmx.compressor.numerical import CastTo
from dmx.compressor.sparse import Sparsify
from .utils import *


class InputOutputTransformer(fx.Transformer):
    """
    A transformer that transforms the module by adding additional ops, which includes:
    - casting
    - approximator
    - sparsifier

    Args:
        module (fx.GraphModule): module to be added with additional ops.
        scopeDict (Optional[dict]): Dictionary that maps node name to scope. Defaults to None.
        cfg (Optional[str]): config file for setting the added ops formats. Defaults to None.

    Attributes:
        module (fx.GraphModule): module to be added with additional ops.
        scopeDict (Optional[str]): Dictionary that maps node name to scope.
        cfg (Optional[str]): config file for setting the added ops formats.

    """

    def __init__(self, module: fx.GraphModule, scopeDict: dict = None, cfg=None):
        super().__init__(module)
        self.scopeDict = scopeDict
        self.config = None
        # if cfg:
        #     self.config = DmxConfig().from_yaml(cfg)
        self.module = module

    def placeholder(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Proxy:
        """
        Execute a ``placeholder`` node. Adds input_cast module after the node.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            A Proxy containing the last node added (input_cast) and the tracer of the new graph
        """
        assert isinstance(target, str)
        placeholder_node = self.new_graph.placeholder(target)
        # Default input cast
        layer = self.scopeDict[placeholder_node.name][0]
        cast_name = target + "_cast"
        cast_format = "SAME"
        # Find input_cast format in cfg if exists
        if self.config:
            layer_key = layer.split("__")[-1]
            if layer_key and layer_key in self.config:
                cast_format = self.config[layer_key]["input_format"]

        self.module.add_submodule(cast_name, CastTo(format=cast_format))
        placeholder_node_cast = self.new_graph.create_node(
            "call_module", cast_name, args=(placeholder_node,)
        )
        return Proxy(placeholder_node_cast, self.tracer)

    def output(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute an ``output`` node. Insert an output_cast module before the node.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            A Proxy containing the last node added (output) and the tracer of the new graph
        """
        output_node = self.new_graph.output(target)
        # Default output cast
        layer = self.scopeDict[output_node.name][0]
        cast_name = target + "_cast"
        cast_format = "SAME"
        # Find output_cast format in cfg if exists
        if self.config:
            layer_key = layer.split("__")[-1]
            if layer_key and layer_key in self.config:
                cast_format = self.config[layer_key]["output_format"]

        self.module.add_submodule(cast_name, CastTo(format=cast_format))
        self.new_graph.inserting_before(output_node)
        output_node_cast = self.new_graph.create_node(
            "call_module", cast_name, args=(output_node.prev,)
        )
        output_node.args = (output_node_cast,)
        self.new_graph.erase_node(output_node)
        return Proxy(output_node_cast, self.tracer)

    def get_attr(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Proxy:
        """
        Execute a ``get_attr`` node. Insert a weight_cast/bias_cast after the node. Insert additional weight_sparsifier node
        if the current node is a weight node and sparsifier format is specified in the cfg file.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns
            A Proxy containing the last node added (output) and the tracer of the new graph
        """
        assert isinstance(target, str)
        get_attr_node = self.new_graph.get_attr(target)
        prev_node = get_attr_node
        # Default cast
        layer = self.scopeDict[get_attr_node.name][0]
        cast_name = target + "_cast"
        cast_format = "SAME"
        # Default sparsifier
        sparsify_format = "DENSE"
        sparsify_name = target + "_sparsifier"
        layer_key = layer.split("__")[-1]
        # Find casting and sparsifier format in config if exists
        if self.config:
            if layer_key and layer_key in self.config:
                if "weight" in target:
                    if "weight_format" in self.config[layer_key]:
                        cast_format = self.config[layer_key]["weight_format"]
                    if "weight_sparseness" in self.config[layer_key]:
                        # Add sparsifier
                        sparsify_format = self.config[layer_key]["weight_sparseness"]
                else:
                    if "bias_format" in self.config[layer_key]:
                        cast_format = self.config[layer_key]["bias_format"]

        self.module.add_submodule(cast_name, CastTo(format=cast_format))

        # Add sparsifier and approximator for weight nodes if needed
        if "weight" in target:
            # Sparsifier submodules needed to be added separately even for default as tensor size
            # is not the same for every layer
            if layer.split("__")[-1] == "model":
                tensor_size = self.module.get_parameter(target).size()
            else:
                tensor_size = (
                    self.module.get_submodule(layer.split("__")[-1])
                    .get_parameter(target.split(".")[-1])
                    .size()
                )
            self.module.add_submodule(
                sparsify_name, Sparsify(tensor_size, sparseness=sparsify_format)
            )
            prev_node = self.new_graph.create_node(
                "call_module", sparsify_name, args=(prev_node,)
            )

            if self.submodules.get("approximator"):
                prev_node = self.new_graph.create_node(
                    "call_module", "approximator", args=(prev_node,)
                )

        get_attr_node_cast = self.new_graph.create_node(
            "call_module", cast_name, args=(prev_node,)
        )
        return Proxy(get_attr_node_cast, self.tracer)

    def call_function(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_function`` node. Insert a cast node after the node.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns
            A Proxy containing the last node added (cast) and the tracer of the new graph
        """
        assert callable(target)

        call_fnc_node = self.new_graph.call_function(target)
        # Observed that inputs to the functions will be wrapped in proxies, parameters of
        # functions is not wrapped in proxies. We need to do a unwrap for proxies before passing to new node.
        call_fnc_node.args = process_args(args)
        call_fnc_node.kwargs = process_kwargs(kwargs)
        # approx_name = call_fnc_node.name + "_approx"
        # self.module.add_submodule(approx_name, Approximator())
        # call_fnc_node_approx = self.new_graph.create_node(
        #     "call_module", approx_name, args=(call_fnc_node,)
        # )
        result_node = call_fnc_node
        layer = self.scopeDict[call_fnc_node.name][0]
        if self.config:
            cast_name = call_fnc_node.name + "_cast"
            layer_key = layer.split("__")[-1]
            if (
                layer_key
                and layer_key in self.config
                and self.config[layer_key]["instance"].lower()
                in call_fnc_node.name.lower()
            ):
                cast_format = self.config[layer_key]["output_format"]
                self.module.add_submodule(cast_name, CastTo(format=cast_format))
                result_node = self.new_graph.create_node(
                    "call_module", cast_name, args=(result_node,)
                )
        return Proxy(result_node, self.tracer)

    def call_method(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_method`` node.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns
            A Proxy containing the call_method node and the tracer of the new graph
        """
        assert isinstance(target, str)
        call_method_node = self.new_graph.call_method(target)
        # Observed that inputs to the functions will be wrapped in proxies, parameters of
        # functions is not wrapped in proxies. We need to do a unwrap for proxies before passing to new node.
        call_method_node.args = process_args(args)
        call_method_node.kwargs = process_kwargs(kwargs)
        return Proxy(call_method_node, self.tracer)
