#!/usr/bin/env python3
import torch
import torch.fx as fx
from torch.fx.node import Argument, Node, Target
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from mltools.functional.approximate import NoApproximation
from mltools import dmx

FORMAT_DICT = {
    "input_cast": "SAME",
    "output_cast": "SAME",
    "residual_cast": "SAME",
    "multiplier_cast": "SAME",
    "accum_cast": "SAME",
    "weight_cast": "SAME",
    "bias_cast": "SAME",
}


class InfoNode:
    """Class for storing information of a node. args and kwargs attributes are used for Graphviz drawing"""

    def __init__(self, name: str, args: List, kwargs: Dict):
        self.name = name
        self.args = args
        self.kwargs = kwargs


class InfoEdge:
    """Class for storing information of an edge. args and kwargs attributes are used for Graphviz drawing"""

    def __init__(self, name: str, args: List, kwargs: Dict):
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.device = None
        self.size = None
        self.dtype = None


class MetadataInterpreter(fx.Interpreter):
    """Interpreter that captures the metadata, eg: device, dtype, shape, of tensors that passed through the Graphmodule
    Example:
        >>> import mltools
        >>> import torch
        >>> net = torch.nn.Sequential(torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,10))
        >>> gm = mltools.fx.transform.substitute_transform(net)
        >>> inp = torch.rand(1,64)
        >>> nodeDict = mltools.fx.transformer.NodeDictTransformer(gm).transform()
        >>> gi = mltools.utils.MetadataInterpreter(gm, nodeDict)
        >>> gi.nodes[0].__dict__ # Check node information
        {'name': '_0',
        'args': ['140117175215088', '_0_0'],
        'kwargs': {'fillcolor': '#D5D6EA', 'shape': 'circle'},
        'input_cast': SAME,
        'output_cast': SAME,
        'accum_cast': SAME,
        'weight_cast': SAME,
        'bias_cast': SAME,
        'weight_sparsifier': DENSE,
        'approximator': NONE}
        >>> e = [e for e in gi.edges]
        >>> e
        ['_0', 'input_1_0', '_1', '_2']
        >>> gi.edges['_0'].__dict__ # Check edge information
        {'name': '_0',
        'args': ['140117175215088',
        '140117175212144',
        '[1, 64]\ntorch.float32\ncpu\n'],
        'kwargs': {'arrowhead': 'open'},
        'size': [1, 64],
        'dtype': torch.float32,
        'device': device(type='cpu')}
    """

    def __init__(self, module: fx.GraphModule, nodeDict: Dict):
        super().__init__(module)

        self.nodeDict = nodeDict
        self.functionNum = 0
        self.methodNum = 0
        self.moduleNum = 0
        # A list that stores node information
        self.nodes = list()
        # A dictionary that stores edge info attached to the node, an edge is stored under the first node it attaches to
        # Both node and edge infos are stored as a tuple of args and kwargs
        self.edges = dict()
        self.input_names = []

    def run(self, *args, initial_env: Optional[Dict[Node, Any]] = None) -> Any:
        self.sizes = dict()
        self.dtypes = dict()
        self.devices = dict()
        self.env = initial_env if initial_env else {}

        # Positional function args are consumed left-to-right by
        # `placeholder` nodes. Use an iterator to keep track of
        # position and extract those values.
        self.args_iter: Iterator[Any] = iter(args)

        for node in self.module.graph.nodes:
            if node in self.env:
                # Short circuit if we have this value. This could
                # be used, for example, for partial evaluation
                # where the caller has pre-populated `env` with
                # values for a subset of the program.
                continue

            self.env[node] = self.run_node(node)
            if isinstance(self.env[node], torch.Tensor):
                self.sizes[node.name] = list(self.env[node].shape)
                self.dtypes[node.name] = self.env[node].dtype
                self.devices[node.name] = self.env[node].device
            elif isinstance(self.env[node], torch.Size):
                self.sizes[node.name] = list(self.env[node])
            if self.garbage_collect_values:
                for to_delete in self.user_to_last_uses.get(node, []):
                    del self.env[to_delete]

            if node.op == "output":
                output_val = self.env[node]
                return output_val

    def run_node(self, n: Node) -> Any:
        return super().run_node(n)

    def process_edge_metadata(self, edge_name: str, node_name: str) -> str:
        """Adds metadata of the edge to InfoEdge and generate the printout of the InfoEdge"""
        print_out = ""
        if node_name in self.sizes:
            self.edges[edge_name].size = self.sizes[node_name]
            print_out += str(self.edges[edge_name].size) + "\n"
        if node_name in self.dtypes:
            self.edges[edge_name].dtype = self.dtypes[node_name]
            print_out += str(self.edges[edge_name].dtype) + "\n"
        if node_name in self.devices:
            self.edges[edge_name].device = self.devices[node_name]
            print_out += str(self.edges[edge_name].device) + "\n"
        return print_out

    def add_edges(self, node: Node) -> None:
        target_name = node.name
        self.edges[target_name] = InfoNode(
            target_name,
            [str(id(target_name)), ""],
            {"arrowhead": "open"},
        )
        for i, argNode in enumerate(node.args):
            if isinstance(argNode, Node):
                # argNode is input
                if argNode.name in self.input_names:
                    name = argNode.name + target_name
                    self.edges[name] = InfoEdge(
                        name,
                        [str(id("start")), str(id(target_name)), argNode.name + ": "],
                        {"fillcolor": "green", "arrowsize": "2"},
                    )

                # argNode is input to multiple nodes
                elif len(self.edges[argNode.name].args) > 2:
                    name = argNode.name + target_name
                    self.edges[name] = InfoEdge(
                        name,
                        [str(id(argNode.name)), str(id(target_name)), ""],
                        {"arrowhead": "open"},
                    )

                else:
                    self.edges[argNode.name].args.insert(1, str(id(target_name)))
                    name = argNode.name
                print_out = self.process_edge_metadata(name, argNode.name)
                self.edges[name].args[-1] += print_out
            else:
                nodeName = repr(argNode) + target_name
                self.nodes.append(
                    InfoNode(
                        nodeName,
                        [str(id(nodeName)) + str(i), repr(argNode)],
                        {"fillcolor": "#D9E3DA", "shape": "oval"},
                    )
                )
                self.edges[nodeName + str(i)] = InfoEdge(
                    nodeName + str(i),
                    [str(id(nodeName)) + str(i), str(id(target_name)), ""],
                    {"arrowhead": "open"},
                )
        for kwargNode in node.kwargs.values():
            if isinstance(kwargNode, Node):
                # kwargNode is input to multiple nodes
                if len(self.edges[kwargNode.name].args) > 2:
                    name = kwargNode.name + target_name
                    self.edges[name] = InfoEdge(
                        name,
                        [str(id(kwargNode.name)), str(id(target_name)), ""],
                        {"arrowhead": "open"},
                    )
                else:
                    self.edges[kwargNode.name].args.insert(1, str(id(target_name)))
                    name = kwargNode.name
                print_out = self.process_edge_metadata(name, kwargNode.name)
                self.edges[name].args[-1] += print_out

    # Main Node running APIs
    def placeholder(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        # Green edge, target could be name
        self.input_names.append(self.nodeDict[target].name)
        return super().placeholder(target, args, kwargs)

    def get_attr(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        # Weight or bias (Square box)
        name = self.nodeDict[target].name
        self.nodes.append(
            InfoNode(name, [(str(id(name))), name], {"fillcolor": "#F9B5AC"})
        )
        self.edges[name] = InfoEdge(name, [str(id(name)), ""], {"arrowhead": "open"})
        return super().get_attr(target, args, kwargs)

    def call_function(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        # Diamond with target inside
        # args are used to put text above the in edges
        # output is out edge
        call_func_node = self.nodeDict["function"][self.functionNum]
        self.functionNum += 1
        target_name = call_func_node.name
        self.nodes.append(
            InfoNode(
                target_name,
                [(str(id(target_name))), target_name],
                {"fillcolor": "#edc485", "shape": "diamond"},
            )
        )
        self.add_edges(call_func_node)
        return super().call_function(target, args, kwargs)

    def call_method(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        # Diamond with target inside (slightly different shade)
        # args are used to put text above the in edges
        # output is out edge
        call_method_node = self.nodeDict["method"][self.methodNum]
        self.methodNum += 1
        target_name = call_method_node.name
        self.nodes.append(
            InfoNode(
                target_name,
                [(str(id(target_name))), target_name],
                {"fillcolor": "#F3DCD4", "shape": "diamond"},
            )
        )
        self.add_edges(call_method_node)
        return super().call_method(target, args, kwargs)

    def call_module(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        # Circle with target inside
        # args are used to put text above the in edges
        # output is out edge

        color = "#D5D6EA"
        call_module_node = self.nodeDict["module"][self.moduleNum]
        self.moduleNum += 1
        target_name = call_module_node.name
        print_out = target_name
        self.nodes.append(
            InfoNode(
                target_name,
                [(str(id(target_name))), ""],
                {"fillcolor": color, "shape": "circle"},
            )
        )
        if isinstance(self.module.get_submodule(target), dmx.nn.DmxModule):
            for ops in FORMAT_DICT:
                cast = getattr(self.module.get_submodule(target), ops)
                if cast:
                    format = cast.format
                    setattr(self.nodes[-1], ops, format)
                    if repr(format) != FORMAT_DICT[ops]:
                        print_out += "\n" + f"{ops}: {repr(format)}"

            sparsifier = self.module.get_submodule(target).weight_sparsifier
            if sparsifier:
                sparseness = sparsifier.sparseness
                setattr(self.nodes[-1], "weight_sparsifier", sparseness)
                if repr(sparseness) != "DENSE":
                    print_out += "\n" + f"weight_sparsifier: {repr(sparseness)}"

            approximator = self.module.get_submodule(target).approximator
            if approximator:
                approx_func = approximator.function
                setattr(self.nodes[-1], "approximator", approx_func)
                if not isinstance(approx_func, NoApproximation):
                    print_out += "\n" + f"approximator: {repr(approx_func)}"
            self.nodes[-1].args[-1] += print_out
        self.add_edges(call_module_node)
        return super().call_module(target, args, kwargs)

    def output(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        # Blue arrow with output text
        output_node = self.nodeDict[target]
        for argNode in output_node.args:
            self.edges[argNode.name].args.insert(1, str(id("end")))
            self.edges[argNode.name].kwargs["arrowhead"] = "normal"
            self.edges[argNode.name].kwargs["arrowsize"] = "2"
            self.edges[argNode.name].kwargs["fillcolor"] = "blue"
            self.edges[argNode.name].args[2] = "output: " + self.process_edge_metadata(
                argNode.name, argNode.name
            )
        return super().output(target, args, kwargs)
