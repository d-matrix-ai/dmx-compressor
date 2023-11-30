from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, Set

import torch
import torch.fx as fx
from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate
from torch.fx.proxy import Proxy
import re

from mltools.numerical import CastTo
from mltools.sparse import Sparsify, Sparseness
from mltools import dmx
import inspect


dmx_aware_mapping = {
    "torch.nn.modules.linear.Linear": dmx.nn.Linear,
    "torch.nn.modules.conv.Conv1d": dmx.nn.Conv1d,
    "torch.nn.modules.conv.Conv2d": dmx.nn.Conv2d,
    "torch.nn.modules.pooling.AdaptiveAvgPool2d": dmx.nn.AdaptiveAvgPool2d,
    "torch.nn.modules.pooling.MaxPool2d": dmx.nn.MaxPool2d,
    "torch.nn.modules.batchnorm.BatchNorm2d": dmx.nn.BatchNorm2d,
    "torch.nn.modules.normalization.LayerNorm": dmx.nn.LayerNorm,
    "torch.nn.modules.dropout.Dropout": dmx.nn.Dropout,
    "torch.nn.modules.activation.Softmax": dmx.nn.Softmax,
    "torch.nn.modules.activation.ReLU": dmx.nn.ReLU,
    "torch.nn.modules.activation.ReLU6": dmx.nn.ReLU6,
    "torch.nn.modules.activation.SiLU": dmx.nn.SiLU,
    "torch.nn.modules.activation.Tanh": dmx.nn.Tanh,
    "torch.nn.modules.activation.GELU": dmx.nn.GELU,
    "transformers.activations.NewGELUActivation": dmx.nn.GELU,
    "transformers.activations.GELUActivation": dmx.nn.GELU,
    "transformers.activations.FastGELUActivation": dmx.nn.GELU,
    "transformers.activations.QuickGELUActivation": dmx.nn.GELU,
    "transformers.activations.ClippedGELUActivation": dmx.nn.GELU,
    "transformers.pytorch_utils.Conv1D": dmx.nn.HFTransformersConv1D,
    "transformers.models.bloom.modeling_bloom.BloomGelu": dmx.nn.GELU,
    "transformers.models.t5.modeling_t5.T5LayerNorm": dmx.nn.HFTransformersT5LayerNorm,
    "transformers.activations.SiLUActivation": dmx.nn.SiLU,
    "transformers.models.llama.modeling_llama.LlamaRMSNorm": dmx.nn.HFTransformersLlamaRMSNorm,
    "transformers.activations.SiLUActivation": dmx.nn.SiLU,
}

dmx_aware_functional_mappings = {
    "torch.nn.functional.relu": dmx.nn.ReLU,
    "torch.nn.functional.relu6": dmx.nn.ReLU6,
    "torch.nn.functional.silu": dmx.nn.SiLU,
    "torch.nn.functional.tanh": dmx.nn.Tanh,
    "torch.nn.functional.gelu": dmx.nn.GELU,
    "torch.nn.functional.softmax": dmx.nn.Softmax,
    "torch.nn.functional.dropout": dmx.nn.Dropout,
    "torch.matmul": dmx.nn.ActActMatMul,
}
for f_key in list(dmx_aware_functional_mappings.keys()):
    new_key = repr(eval(f_key))
    dmx_aware_functional_mappings[new_key] = dmx_aware_functional_mappings.pop(f_key)
dmx_aware_functional_mappings["<built-in function add>"] = dmx.nn.ResAdd


def process_args(args):
    """
    A function that goes through each arg in args and removes the proxy wrapper.

    Args:
        args (Tuple): args for which proxy wrappers should be removed

    Returns:
        A tuple of args where each element has the proxy wrapper removed

    """
    new_args = ()
    for arg in args:
        if isinstance(arg, Proxy):
            new_args += (arg.node,)
        else:
            new_args += (arg,)
    return new_args


def process_kwargs(kwargs):
    """
    A function that goes through each kwarg in kwargs and removes the proxy wrapper.

    Args:
        kwargs (Dict): kwargs for which proxy wrappers should be removed

    Returns:
        A dictionary of kwargs where each element has the proxy wrapper removed

    """
    new_kwargs = dict()
    for k in kwargs.keys():
        if isinstance(kwargs[k], Proxy):
            new_kwargs[k] = kwargs[k].node
        else:
            new_kwargs[k] = kwargs[k]
    return new_kwargs


def get_name_for_func_nodes(
    candidate: str, used_names: Set[str], base_count: Dict[str, int]
):
    """Get the unique name for functional nodes

    Arguments:
        candidate (str): used as the basis for the unique name, relevant to the user.
        used_names (Set[str]): A Set of names already used for nodes
        base_count (Dict[str, int]): A dict counting number of names sharing the same candidate base
    """
    illegal_char_regex = re.compile("[^0-9a-zA-Z_]+")
    name_suffix_regex = re.compile(r"(.*)_(\d+)$")

    candidate = illegal_char_regex.sub("_", candidate)
    if not candidate:
        candidate = "_unnamed"

    if candidate[0].isdigit():
        candidate = f"_{candidate}"

    match = name_suffix_regex.match(candidate)
    if match is None:
        base = candidate
        num = None
    else:
        base, num_str = match.group(1, 2)
        num = int(num_str)

    candidate = base if num is None else f"{base}_{num}"
    if not num:
        num = base_count[base]

    while candidate in used_names or fx.graph._Namespace()._is_illegal_name(
        candidate, None
    ):
        num += 1
        candidate = f"{base}_{num}"
    return candidate


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
    """

    def __init__(self, module: fx.GraphModule, node_name_to_scope: dict):
        super().__init__(module)
        self.module = module
        self.node_name_to_scope = node_name_to_scope

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
        self.module.add_submodule(
            target, dmx_aware_mapping[node_key].from_raw(curr_mod)
        )
        new_node = self.new_graph.create_node(
            "call_module", target, args=(args[0].node,)
        )
        return Proxy(new_node, self.tracer)

    def create_unique_name_in_scope(self, cand_name):
        curr_name = get_name_for_func_nodes(
            cand_name,
            self.new_graph._graph_namespace._used_names,
            self.new_graph._graph_namespace._base_count,
        )
        # replace "_" with "." exit for last "_" if new_name ends with digit
        new_name = curr_name.replace("_", ".")
        new_name = (
            new_name[: new_name.rfind(".")] + "_" + new_name[new_name.rfind(".") + 1 :]
            if new_name[-1].isdigit()
            else new_name
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
        # Skip tranformation for add that is not in a resnet
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
                and args[0].node.op == "call_module"
                and args[1].node.op == "call_module"
            ):
                cand_name = curr_target + ".resadd"
                new_name = self.create_unique_name_in_scope(cand_name)
            else:
                return super().call_function(target, args, kwargs)
        elif node_key == repr(eval("torch.matmul")):
            cand_name = curr_target + ".matmul"
            new_name = self.create_unique_name_in_scope(cand_name)
        else:
            new_name = curr_target + "." + candidate if curr_target != "" else candidate
        # If new name is not candidate, need to add candidate to used names,
        # otherwise next call_function will use the same candidate. (create_name is also called in create_node)
        if new_name != candidate:
            self.new_graph._graph_namespace.create_name(candidate, None)
        # find out what kwargs to pass in to new module
        empty_mod = dmx_aware_functional_mappings[node_key]()
        accepted_kwarg_keys = inspect.signature(empty_mod.__init__).parameters.keys()
        newkwargs = {}
        for key, value in kwargs.items():
            if key in accepted_kwarg_keys:
                newkwargs[key] = value
        self.module.add_submodule(
            new_name, dmx_aware_functional_mappings[node_key](**newkwargs)
        )
        new_node = self.new_graph.create_node(
            "call_module",
            new_name,
        )
        new_node.args = process_args(args)
        return Proxy(new_node, self.tracer)


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
