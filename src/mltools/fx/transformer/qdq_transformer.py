#!/usr/bin/env python3
import copy

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, Set

import torch
import torch.fx as fx
from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate
from torch.fx import Graph, Proxy
import torch.fx.traceback as fx_traceback

import itertools

from mltools import dmx
from mltools.fx.transformer.utils import process_args
from mltools.numerical import Quantize, DeQuantize

from .utils import dmx_aware_mapping, dmx_aware_functional_mappings


class QdQTransformer(fx.Transformer):
    def __init__(self, module: fx.GraphModule, scopeDict: dict = None, cfg=None):
        super().__init__(module)
        self.scopeDict = scopeDict
        self.config = None
        self.module = module
        self.module.recompile()

    @staticmethod
    def substitute_compiler_graph(
        og: "Graph",
        g: "Graph",
        val_map: Dict[Node, Node],
        target_prefix: str,
        curmod: torch.nn.Module,
        rootmod: torch.nn.Module,
        return_output_node=False,
    ) -> "Optional[Argument]":
        def my_node_copy(
            g: Graph,
            node: Node,
            curmod: torch.nn.Module,
            rootmod: torch.nn.Module,
            target_prefix: str,
            arg_transform: Callable[[Node], "Argument"] = lambda x: x,
        ) -> Node:
            """ """
            args = map_arg(node.args, arg_transform)
            kwargs = map_arg(node.kwargs, arg_transform)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)

            if node.op == "get_attr":
                # Fixup the target by adding in the module prefix
                curnode_target = target_prefix + "." + node.target
            else:
                curnode_target = node.target

            result_node = g.create_node(
                node.op,
                curnode_target,
                args,
                kwargs,
                node.name,
                node.type,
            )
            result_node.meta = copy.copy(node.meta)
            return result_node

        for node in g.nodes:
            if node in val_map:
                continue
            if node.op == "output":
                rv = map_arg(node.args[0], lambda n: val_map[n])
                return rv if not return_output_node else (rv, node)
            val_map[node] = my_node_copy(og, node, curmod, rootmod, target_prefix, lambda n: val_map[n])
        return None

    def call_module(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """ Check if the current module that 'target' points to is in the dmx_aware_mapping
            and substitutes in the fxir subgraph
        """
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        curr_mod = submod

        dmx_module_targets = itertools.chain(dmx_aware_mapping.values(),
                                             dmx_aware_functional_mappings.values())
        if any(map(lambda m: isinstance(curr_mod, m), dmx_module_targets)):
            inv_dmx_aware_mapping = {str(v): k for k, v in dmx_aware_mapping.items()}
            subgraph = curr_mod.to_compiler_graph()
            processed_args = process_args(args)
            subgraph_input_nodes = filter(lambda n: n.op == "placeholder", list(subgraph.nodes))
            val_map = {n: processed_args[i] for i, n in enumerate(subgraph_input_nodes)}
            print(val_map)
            curr_node = self.substitute_compiler_graph(
                self.new_graph, subgraph, val_map, target, curr_mod, self.module, False
            )
            if isinstance(curr_node, torch.fx.node.Node):
                curr_node = Proxy(curr_node)
            return curr_node
        else:
            curr_node = self.tracer.call_module(submod, submod.forward, args, kwargs)
            return curr_node

    def transform(self) -> fx.GraphModule:
        with fx_traceback.preserve_node_meta():
            result = super().run(enable_io_processing=False)
        if result is not None:
            def strip_proxy(a : Union[Argument, Proxy]) -> Any:
                return a.node if isinstance(a, Proxy) else a
            self.new_graph.output(map_aggregate(result, strip_proxy))
        return fx.GraphModule(self.module, self.new_graph)
