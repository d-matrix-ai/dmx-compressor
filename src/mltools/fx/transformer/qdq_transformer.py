#!/usr/bin/env python3
import copy

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, Set

import torch
import torch.fx as fx
from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate
from torch.fx import Graph, Proxy
import torch.fx.traceback as fx_traceback

from mltools import dmx
from mltools.fx.transformer.utils import process_args
from mltools.numerical import Quantize, DeQuantize


class QdQTransformer(fx.Transformer):
    def __init__(self, module: fx.GraphModule, scopeDict: dict = None, cfg=None):
        super().__init__(module)
        self.scopeDict = scopeDict
        self.config = None
        self.module = module
        self.module.recompile()

    @staticmethod
    def my_graph_copy(
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
            arg_transform: Callable[[Node], "Argument"] = lambda x: x,
        ) -> Node:
            """ """
            args = map_arg(node.args, arg_transform)
            kwargs = map_arg(node.kwargs, arg_transform)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)

            if node.op == "get_attr" or node.op == "call_function":
                if node.op == "get_attr":
                    curnode_target = target_prefix + "." + node.target
                    curnode_target_q = curnode_target + "_q"
                    curnode_target_dq = curnode_target + "_dq"
                elif node.op == "call_function":
                    curnode_target = node.target
                    curnode_target_q = target_prefix + ".linear_q"
                    curnode_target_dq = target_prefix + ".linear_dq"

                # TODO plumb in the original scale, zero_point, dtype
                quantize = Quantize(1, 0, torch.float16)
                rootmod.add_submodule(curnode_target_q, quantize)

                dequantize = DeQuantize(1, 0, torch.float16)
                rootmod.add_submodule(curnode_target_dq, dequantize)

                result_node = g.create_node(
                    node.op,
                    curnode_target,
                    args,
                    kwargs,
                    node.name,
                    node.type,
                )

                result_node = g.create_node(
                    'call_module',
                    curnode_target_q,
                    (result_node,),
                )

                result_node = g.create_node(
                    'call_module',
                    curnode_target_dq,
                    (result_node,),
                )
            else:
                result_node = g.create_node(
                    node.op, node.target, args, kwargs, node.name, node.type
                )
            result_node.meta = copy.copy(node.meta)
            return result_node

        for node in g.nodes:
            if node in val_map:
                continue
            if node.op == "output":
                rv = map_arg(node.args[0], lambda n: val_map[n])
                return rv if not return_output_node else (rv, node)
            val_map[node] = my_node_copy(og, node, curmod, rootmod, lambda n: val_map[n])
        return None

    def call_module(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:

        # TODO Move this to a better location
        def lift_node(p):
            if not isinstance(p, Proxy):
                return p
            else:
                return lift_node(p.node)

        assert isinstance(target, str)

        submod = self.fetch_attr(target)
        curr_mod = submod

        if isinstance(curr_mod, dmx.nn.Linear):
            new_submod, subgraph = curr_mod.to_compiler_graph()
            input_node = list(subgraph.nodes)[0]
            val_map = {
                input_node: process_args(args)[0],
            }
            curr_node = self.my_graph_copy(
                self.new_graph, subgraph, val_map, target, curr_mod, self.module, False
            )
            return curr_node

        # elif isinstance(curr_mod, dmx.nn.ResAdd):
        #     new_submod, subgraph = curr_mod.to_compiler_graph()
        #     input_node = list(subgraph.nodes)[0]
        #     val_map = {
        #         input_node: process_args(args)[0],
        #     }
        #     curr_node = self.my_graph_copy(
        #         self.new_graph, subgraph, val_map, target, curr_mod, self.module, False
        #     )
        #     import ipdb; ipdb.set_trace()
        #     return curr_node
        else:
            curr_node = self.tracer.call_module(submod, submod.forward, args, kwargs)
            return curr_node

    def transform(self) -> fx.GraphModule:
        """
        Transform ``self.module`` and return the transformed
        ``GraphModule``.
        """
        with fx_traceback.preserve_node_meta():
            result = super().run(enable_io_processing=False)
        if result is not None:
            def strip_proxy(a : Union[Argument, Proxy]) -> Any:
                return a.node if isinstance(a, Proxy) else a
            self.new_graph.output(map_aggregate(result, strip_proxy))
        return fx.GraphModule(self.module, self.new_graph)
