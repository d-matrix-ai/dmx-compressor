import torch.fx as fx
from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate
from torch.fx.proxy import Proxy
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from mltools.numerical import CastTo
from mltools.sparse import Sparsify, Sparseness
from mltools.approximate import Approximator
from mltools.corsair import CorsairConfig


class InputOutputTransformer(fx.Transformer):
    """
    A transformer that transforms the module by adding additional ops, which includes:
    - casting
    - approximator
    - sparsifier
    """

    def __init__(self, module: fx.GraphModule, scopeDict: dict = None, cfg=None):
        super().__init__(module)
        self.scopeDict = scopeDict
        self.config = None
        if cfg:
            self.config = CorsairConfig().from_yaml(cfg)
        self.module = module

    def placeholder(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Proxy:
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
        assert callable(target)

        call_fnc_node = self.new_graph.call_function(target)
        # Observed that inputs to the functions will be wrapped in proxies, parameters of
        # functions is not wrapped in proxies. We need to do a unwrap for proxies before passing to new node.
        new_kwargs = dict()
        for k in kwargs.keys():
            if isinstance(kwargs[k], Proxy):
                new_kwargs[k] = kwargs[k].node
            else:
                new_kwargs[k] = kwargs[k]
        new_args = ()
        for arg in args:
            if isinstance(arg, Proxy):
                new_args += (arg.node,)
            else:
                new_args += (arg,)
        call_fnc_node.args = new_args
        call_fnc_node.kwargs = new_kwargs
        approx_name = call_fnc_node.name + "_approx"
        self.module.add_submodule(approx_name, Approximator())
        call_fnc_node_approx = self.new_graph.create_node(
            "call_module", approx_name, args=(call_fnc_node,)
        )
        result_node = call_fnc_node_approx
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
                    "call_module", cast_name, args=(call_fnc_node_approx,)
                )
        return Proxy(result_node, self.tracer)

    def call_method(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        assert isinstance(target, str)
        call_method_node = self.new_graph.call_method(target)
        # Observed that inputs to the functions will be wrapped in proxies, parameters of
        # functions is not wrapped in proxies. We need to do a unwrap for proxies before passing to new node.
        new_kwargs = dict()
        for k in kwargs.keys():
            if isinstance(kwargs[k], Proxy):
                new_kwargs[k] = kwargs[k].node
            else:
                new_kwargs[k] = kwargs[k]
        new_args = ()
        for arg in args:
            if isinstance(arg, Proxy):
                new_args += (arg.node,)
            else:
                new_args += (arg,)
        call_method_node.args = new_args
        call_method_node.kwargs = new_kwargs

        # cast_name = target+"_cast"
        # cast_format = "SAME"
        # # Not so sure about this part as no cfgs seen for call_method yet
        # # if self.config:
        # #     layer = self.scopeDict[call_method_node.name][0]
        # #     layer_key = layer.split('__')[-1]
        # #     if layer_key and layer_key in self.config:
        # #         cast_format = self.config[layer_key]['output_format']
        # self.module.add_submodule(cast_name,CastTo(format=cast_format))
        # call_method_cast_node = self.new_graph.create_node(
        #     "call_module", cast_name, args=(call_method_node,)
        # )
        # self.nodeDict[call_method_cast_node.target] = call_method_cast_node
        return Proxy(call_method_node, self.tracer)


class NodeDictTransformer(fx.Transformer):
    """
    A transformer that creates a dict contaning mapping between target of node and the node itself
    """

    def __init__(self, module: fx.GraphModule):
        super().__init__(module)
        self.module = module
        # A dictionary to map node.target to node instance
        self.nodeDict = dict()
        self.nodeDict["function"] = list()
        self.nodeDict["method"] = list()

    def placeholder(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Proxy:
        assert isinstance(target, str)
        placeholder_node = self.new_graph.placeholder(target)
        self.nodeDict[placeholder_node.target] = placeholder_node
        return Proxy(placeholder_node, self.tracer)

    def output(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        assert isinstance(target, str)
        output_node = self.new_graph.output(target)
        output_node.args = (output_node.prev,)
        self.nodeDict[output_node.target] = output_node
        return Proxy(output_node, self.tracer)

    def get_attr(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Proxy:
        assert isinstance(target, str)
        get_attr_node = self.new_graph.get_attr(target)
        self.nodeDict[get_attr_node.target] = get_attr_node
        return Proxy(get_attr_node, self.tracer)

    def call_function(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        assert callable(target)
        call_fnc_node = self.new_graph.call_function(target)
        new_kwargs = dict()
        for k in kwargs.keys():
            if isinstance(kwargs[k], Proxy):
                new_kwargs[k] = kwargs[k].node
            else:
                new_kwargs[k] = kwargs[k]
        new_args = ()
        for arg in args:
            if isinstance(arg, Proxy):
                new_args += (arg.node,)
            else:
                new_args += (arg,)
        call_fnc_node.args = new_args
        call_fnc_node.kwargs = new_kwargs
        self.nodeDict["function"].append(call_fnc_node)
        return Proxy(call_fnc_node, self.tracer)

    def call_method(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        assert isinstance(target, str)
        call_method_node = self.new_graph.call_method(target)
        new_kwargs = dict()
        for k in kwargs.keys():
            if isinstance(kwargs[k], Proxy):
                new_kwargs[k] = kwargs[k].node
            else:
                new_kwargs[k] = kwargs[k]
        new_args = ()
        for arg in args:
            if isinstance(arg, Proxy):
                new_args += (arg.node,)
            else:
                new_args += (arg,)
        call_method_node.args = new_args
        call_method_node.kwargs = new_kwargs
        self.nodeDict["method"].append(call_method_node)
        return Proxy(call_method_node, self.tracer)

    def call_module(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        assert isinstance(target, str)
        call_module_node = self.new_graph.call_module(target)
        new_kwargs = dict()
        for k in kwargs.keys():
            if isinstance(kwargs[k], Proxy):
                new_kwargs[k] = kwargs[k].node
            else:
                new_kwargs[k] = kwargs[k]
        new_args = ()
        for arg in args:
            if isinstance(arg, Proxy):
                new_args += (arg.node,)
            else:
                new_args += (arg,)
        call_module_node.args = new_args
        call_module_node.kwargs = new_kwargs
        self.nodeDict[target] = call_module_node
        return Proxy(call_module_node, self.tracer)

    def transform(self) -> dict:
        result = super().run()
        if result is not None:

            def strip_proxy(a: Union[Argument, Proxy]) -> Any:
                return a.node if isinstance(a, Proxy) else a

            self.new_graph.output(map_aggregate(result, strip_proxy))
        return self.nodeDict


class ConfigurationTransformer(fx.Transformer):
    """
    A transformer that changes the format of the ops according to the cfg file
    """

    def __init__(self, module: fx.GraphModule, scopeDict: dict, cfg=None):
        super().__init__(module)
        self.config = None
        if cfg:
            self.config = CorsairConfig().from_yaml(cfg)
        self.module = module
        self.scopeDict = scopeDict

    def placeholder(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Proxy:
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
                            self.module.get_submodule(
                                sparsify_name
                            ).sparseness = Sparseness.from_shorthand(sparsify_format)
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
        Transform ``self.module`` and return the transformed
        ``GraphModule``.
        """
        result = super().run()
        if result is not None:

            def strip_proxy(a: Union[Argument, Proxy]) -> Any:
                return a.node if isinstance(a, Proxy) else a

            self.new_graph.output(map_aggregate(result, strip_proxy))
        return self.module

    def has_module(self, target: str) -> bool:
        for name, _ in self.module.named_modules():
            if name == target:
                return True
        return False
