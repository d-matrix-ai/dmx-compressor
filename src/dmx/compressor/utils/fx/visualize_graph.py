#!/usr/bin/env python3
import torch
import torch.fx as fx
from typing import Any
from graphviz import Digraph
from dmx.compressor.fx import NodeDictTransformer, QuantTracer
from dmx.compressor.utils.fx.interpreter import MetadataInterpreter


def visualize_graph(
    model: torch.nn.Module, input: torch.Tensor, file_name="graph", tracer=QuantTracer()
) -> Any:
    """Saves the graph to file_name if given and returns the pygraph object
    Example use cases:
    visualizing non fx transformed:
        visualize_graph(torch.nn.Linear(64,64),torch.rand(1,64))
    visualizing fx transformed:
        net = DmxModel.from_torch(torch.nn.Linear(64,64))
        visualize_graph(net,torch.rand(1,64))
    Note: make sure inputs follow the same order of kwargs as the module signature, remember to install pygraphviz by sudo apt install graphviz
    """
    if not isinstance(model, fx.GraphModule):
        graph = tracer.trace(model)
        model = fx.GraphModule(tracer.root, graph)
    nodeDict = NodeDictTransformer(model).transform()
    gi = MetadataInterpreter(model, nodeDict)
    if isinstance(input, tuple):
        gi.run(*input)
    else:
        gi.run(input)
    node_attr = dict(
        style="filled",
        shape="box",
        align="center",
        fontsize="12",
        height="0.1",
        fontname="monospace",
    )
    pygraph = Digraph(
        node_attr=node_attr,
        graph_attr=dict(layout="dot"),
        edge_attr=dict(penwidth="2", fontsize="15"),
    )
    pygraph.node(str(id("start")), label="start", fillcolor="white")
    pygraph.node(str(id("end")), label="end", fillcolor="white")
    for info_nodes in gi.nodes:
        pygraph.node(*(info_nodes.args), **(info_nodes.kwargs))
    for info_eges in gi.edges.values():
        if len(info_eges.args) == 3:
            pygraph.edge(*(info_eges.args), **(info_eges.kwargs))

    try:
        pygraph.render(filename=file_name)
    except RuntimeError:
        print(
            "Pygraphviz not installed from root! Please install by 'sudo apt install graphviz'"
        )
