#!/usr/bin/env python3

import torch.fx as fx
from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate
from torch.fx.proxy import Proxy
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union


class InputOutputTransformer(fx.Transformer):
    def placeholder(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Proxy:
        assert isinstance(target, str)
        # TODO for newer versions of fx you can pass in default values
        # default_value = next(iter(args)) if args else inspect.Signature.empty

        placeholder_node = self.new_graph.placeholder(target)
        placeholder_node_cast = self.new_graph.create_node(
            "call_module", "input_cast", args=(placeholder_node,)
        )
        return Proxy(placeholder_node_cast, self.tracer)

    def output(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        output_node = self.new_graph.output(target)
        self.new_graph.inserting_before(output_node)
        output_node_cast = self.new_graph.create_node(
            "call_module", "output_cast", args=(output_node.prev,)
        )
        self.new_graph.erase_node(output_node)
        return Proxy(output_node_cast, self.tracer)

    def get_attr(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Proxy:
        get_attr_node = self.new_graph.get_attr(target)
        get_attr_node_cast = self.new_graph.create_node(
            "call_module", "weight_cast", args=(get_attr_node,)
        )
        return Proxy(get_attr_node_cast, self.tracer)
