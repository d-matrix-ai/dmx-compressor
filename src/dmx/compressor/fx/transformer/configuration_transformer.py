from typing import Any, Dict, Tuple, Union
import torch.fx as fx
from torch.fx.node import Argument, Target, map_aggregate
from torch.fx.proxy import Proxy

from dmx.compressor.sparse import Sparseness


class ConfigurationTransformer(fx.Transformer):
    """
    A transformer that changes the format of the ops according to the cfg file

    Args:
        module (fx.GraphModule): module to be added with additional ops.
        scopeDict (Optional[dict]): Dictionary that maps node name to scope. Defaults to None.
        cfg (Optional[str]): config file for setting the added ops formats. Defaults to None.

    Attributes:
        module (fx.GraphModule): module to be added with additional ops.
        scopeDict (Optional[str]): Dictionary that maps node name to scope.
        cfg (Optional[str]): config file for setting the added ops formats.
    """

    def __init__(self, module: fx.GraphModule, scopeDict: dict, cfg=None):
        super().__init__(module)
        self.config = None
        # if cfg:
        #     self.config = DmxConfig().from_yaml(cfg)
        self.module = module
        self.scopeDict = scopeDict

    def placeholder(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Proxy:
        """
        Execute a ``placeholder`` node. Modifies the format of input_cast according to cfg

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            A Proxy containing the node and the tracer of the new graph
        """
        placeholder_node = self.new_graph.placeholder(target)
        layer = self.scopeDict[placeholder_node.name][0]
        cast_name = target + "_cast"
        if self.config and self.has_module(cast_name):
            layer_key = layer.split("__")[-1]
            if layer_key and layer_key in self.config:
                cast_format = self.config[layer_key]["input_format"]
                self.module.get_submodule(cast_name).set_format(cast_format)
        return super().placeholder(target, args, kwargs)

    def output(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute a ``output`` node. Modifies the format of output_cast according to cfg

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            A Proxy containing the node and the tracer of the new graph
        """
        output_node = self.new_graph.output(target)
        layer = self.scopeDict[output_node.name][0]
        cast_name = target + "_cast"
        if self.config and self.has_module(cast_name):
            layer_key = layer.split("__")[-1]
            if layer_key and layer_key in self.config:
                cast_format = self.config[layer_key]["output_format"]
                self.module.get_submodule(cast_name).set_format(cast_format)
        return super().output(target, args, kwargs)

    def get_attr(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Proxy:
        """
        Execute a ``get_attr`` node. Modifies the format of cast/sparsifier according to cfg

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            A Proxy containing the node and the tracer of the new graph
        """
        get_attr_node = self.new_graph.get_attr(target)
        if get_attr_node.name in self.scopeDict:
            layer = self.scopeDict[get_attr_node.name][0]
            cast_name = target + "_cast"
            sparsify_name = target + "_sparsifier"
            layer_key = layer.split("__")[-1]
            if self.config:
                if layer_key and layer_key in self.config:
                    if "weight" in target:
                        if "weight_format" in self.config[
                            layer_key
                        ] and self.has_module(cast_name):
                            cast_format = self.config[layer_key]["weight_format"]
                            self.module.get_submodule(cast_name).set_format(cast_format)
                        if "weight_sparseness" in self.config[
                            layer_key
                        ] and self.has_module(sparsify_name):
                            sparsify_format = self.config[layer_key][
                                "weight_sparseness"
                            ]
                            self.module.get_submodule(sparsify_name).sparseness = (
                                Sparseness.from_shorthand(sparsify_format)
                            )
                    else:
                        if "bias_format" in self.config[layer_key] and self.has_module(
                            cast_name
                        ):
                            cast_format = self.config[layer_key]["bias_format"]
                            self.module.get_submodule(cast_name).set_format(cast_format)
        return super().get_attr(target, args, kwargs)

    def call_function(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_function`` node. Modifies the format of cast according to cfg

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            A Proxy containing the node and the tracer of the new graph
        """
        call_fnc_node = self.new_graph.call_function(target)
        if self.config and call_fnc_node.name in self.scopeDict:
            layer = self.scopeDict[call_fnc_node.name][0]
            cast_name = call_fnc_node.name + "_cast"
            layer_key = layer.split("__")[-1]
            if (
                layer_key
                and layer_key in self.config
                and self.config[layer_key]["instance"].lower()
                in call_fnc_node.name.lower()
                and self.has_module(cast_name)
            ):
                cast_format = self.config[layer_key]["output_format"]
                self.module.get_submodule(cast_name).set_format(cast_format)
        return super().call_function(target, args, kwargs)

    def transform(self):
        """
        Transform ``self.module`` and return the transformed``GraphModule``.

        Returns:
            A GraphModule with updated ops formats.
        """
        result = super().run()
        if result is not None:

            def strip_proxy(a: Union[Argument, Proxy]) -> Any:
                return a.node if isinstance(a, Proxy) else a

            self.new_graph.output(map_aggregate(result, strip_proxy))
        return self.module

    def has_module(self, target: str) -> bool:
        """
        A function that checks if self.module contains a specific target

        Args:
            target (str): the target to look up

        Returns:
            True if self.module contains a node with the same target
        """
        for name, _ in self.module.named_modules():
            if name == target:
                return True
        return False
