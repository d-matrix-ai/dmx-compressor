#!/usr/bin/env python3
import torch
import torch.fx as fx
from torch.fx.node import Argument, Node, Target
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from graphviz import Digraph
from mltools.fx.transformer import NodeDictTransformer
from mltools.fx.tracer import QuantTracer


class GraphvizInterpreter(fx.Interpreter):
    """Interpreter that draws the graph of a GraphModule"""
    def __init__(self,module:fx.GraphModule,nodeDict:Dict):
        super().__init__(module)
        node_attr = dict(style='filled',
                     shape='box',
                     align='center',
                     fontsize='12',
                     height='0.1',
                     fontname='monospace')
        self.nodeDict = nodeDict
        self.functionNum = 0
        self.methodNum=0
        self.pygraph = Digraph(node_attr = node_attr, graph_attr = dict(layout = "dot"),edge_attr=dict(penwidth = "2",fontsize = "15"))
        # Nodes indicating start and end of the graph
        self.pygraph.node(str(id("start")),label = "start",fillcolor = "white")
        self.pygraph.node(str(id("end")),label = "end",fillcolor = "white")
        # A list that stores node information
        self.nodes = list()
        # A dictionary that stores edge info attached to the node, an edge is stored under the first node it attaches to
        # Both node and edge infos are stored as a tuple of args and kwargs
        self.edges = dict()
        self.sizeDict = dict()

    def run(self, *args, initial_env : Optional[Dict[Node, Any]] = None) -> Any:
        self.sizes=dict()
        self.env = initial_env if initial_env else {}

        # Positional function args are consumed left-to-right by
        # `placeholder` nodes. Use an iterator to keep track of
        # position and extract those values.
        self.args_iter : Iterator[Any] = iter(args)

        for node in self.module.graph.nodes:
            if node in self.env:
                # Short circuit if we have this value. This could
                # be used, for example, for partial evaluation
                # where the caller has pre-populated `env` with
                # values for a subset of the program.
                continue
            
            self.env[node] = self.run_node(node)
            if isinstance(self.env[node],torch.Tensor): 
                self.sizes[node.name] = list(self.env[node].shape) 
            elif isinstance(self.env[node],torch.Size):
                self.sizes[node.name] = list(self.env[node])
            if self.garbage_collect_values:
                for to_delete in self.user_to_last_uses.get(node, []):
                    del self.env[to_delete]

            if node.op == 'output':
                output_val = self.env[node]
                return output_val

    def run_node(self, n : Node) -> Any:
        return super().run_node(n)

    # Main Node running APIs
    def placeholder(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        # Green edge, target could be name
        self.input_name = self.nodeDict[target].name
        return super().placeholder(target,args,kwargs)

    def get_attr(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        # Weight or bias (Square box)
        name = self.nodeDict[target].name
        self.nodes.append(([(str(id(name))),name],{"fillcolor":"#F9B5AC"}))
        self.edges[name] = ([str(id(name)),""],{"arrowhead":"open"})
        return super().get_attr(target,args,kwargs)

    def call_function(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        # Diamond with target inside
        # args are used to put text above the in edges
        # output is out edge
        call_func_node = self.nodeDict["function"][self.functionNum]
        self.functionNum+=1
        target_name = call_func_node.name
        self.nodes.append(([(str(id(target_name))),target_name],{"fillcolor":"#edc485","shape":"diamond"}))
        self.edges[target_name] = ([str(id(target_name)),""],{"arrowhead":"open"})
        for i, argNode in enumerate(call_func_node.args):
            if isinstance(argNode,Node):
                # argNode is input
                if argNode.name == self.input_name:
                    self.edges[self.input_name+target_name] = ([str(id("start")),str(id(target_name)),"input: "],{"fillcolor":"green","arrowsize":"2"})
                    name = argNode.name+target_name
                # argNode is input to multiple nodes
                elif len(self.edges[argNode.name][0])>2:
                    self.edges[argNode.name+target_name] = ([str(id(argNode.name)),str(id(target_name)),""],{"arrowhead":"open"})
                    name = argNode.name+target_name
                else:
                    self.edges[argNode.name][0].insert(1,str(id(target_name)))
                    name = argNode.name
                if argNode.name in self.sizes:
                    self.edges[name][0][-1]+=str(self.sizes[argNode.name])
            else:
                nodeName = repr(argNode)+target_name
                self.nodes.append(([str(id(nodeName))+str(i),repr(argNode)],{"fillcolor":"#D9E3DA","shape":"oval"}))
                self.edges[nodeName+str(i)] = ([str(id(nodeName))+str(i),str(id(target_name)),""],{"arrowhead":"open"})
        for kwargNode in call_func_node.kwargs.values():
            if isinstance(kwargNode,Node):
                # kwargNode is input to multiple nodes
                if len(self.edges[kwargNode.name][0])>2:
                    self.edges[kwargNode.name+target_name] = ([str(id(kwargNode.name)),str(id(target_name)),""],{"arrowhead":"open"})
                    name = kwargNode.name+target_name
                else:
                    self.edges[kwargNode.name][0].insert(1,str(id(target_name)))
                    name = kwargNode.name
                if kwargNode.name in self.sizes:
                    self.edges[name][0][-1]+=str(self.sizes[kwargNode.name])
        return super().call_function(target, args, kwargs)

    def call_method(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        # Diamond with target inside (slightly different shade)
        # args are used to put text above the in edges
        # output is out edge
        call_method_node = self.nodeDict["method"][self.methodNum]
        self.methodNum+=1
        target_name = call_method_node.name
        self.nodes.append(([(str(id(target_name))),target_name],{"fillcolor":"#F3DCD4","shape":"diamond"}))
        self.edges[target_name] = ([str(id(target_name)),""],{"arrowhead":"open"})
        for i,argNode in enumerate(call_method_node.args):
            if isinstance(argNode,Node):
                # argNode is input
                if argNode.name == self.input_name:
                    self.edges[self.input_name+target_name] = ([str(id("start")),str(id(target_name)),"input: "],{"fillcolor":"green","arrowsize":"2"})
                    name = argNode.name+target_name
                # argNode is input to multiple nodes
                elif len(self.edges[argNode.name][0])>2:
                    self.edges[argNode.name+target_name] = ([str(id(argNode.name)),str(id(target_name)),""],{"arrowhead":"open"})
                    name = argNode.name+target_name
                else:
                    self.edges[argNode.name][0].insert(1,str(id(target_name)))
                    name = argNode.name
                if argNode.name in self.sizes:
                    self.edges[name][0][-1]+=str(self.sizes[argNode.name])
            # if argNode is not a fx.node but a normal parameter passed to the method
            else:
                nodeName = repr(argNode)+target_name
                self.nodes.append(([str(id(nodeName))+str(i),repr(argNode)],{"fillcolor":"#D9E3DA","shape":"oval"}))
                self.edges[nodeName+str(i)] = ([str(id(nodeName))+str(i),str(id(target_name)),""],{"arrowhead":"open"})
        for kwargNode in call_method_node.kwargs.values():
            if isinstance(kwargNode,Node):
                # kwargNode is input to multiple nodes
                if len(self.edges[kwargNode.name][0])>2:
                    self.edges[kwargNode.name+target_name] = ([str(id(kwargNode.name)),str(id(target_name)),""],{"arrowhead":"open"})
                    name = kwargNode.name+target_name
                else:
                    self.edges[kwargNode.name][0].insert(1,str(id(target_name)))
                    name = kwargNode.name
                if kwargNode.name in self.sizes:
                    self.edges[name][0][-1]+=str(self.sizes[kwargNode.name])

        return super().call_method(target,args,kwargs)

    def call_module(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        # Circle with target inside (different shape for different casts)
        # args are used to put text above the in edges
        # output is out edge
        format = ""
        if "cast" in target:
            color = "#9BB8ED"
            format = "\n\n" + repr(self.module.get_submodule(target).format)
        elif "spars" in target:
            color = "#A39FE1"
            format = "\n\n" + repr(self.module.get_submodule(target).sparseness)
        elif "approx" in target:
            color = "#DCD0EA"
            format = "\n\n" + repr(self.module.get_submodule(target).function)
        else:
            color = "#D5D6EA"
        call_module_node = self.nodeDict[target]
        target_name = call_module_node.name
        self.nodes.append(([(str(id(call_module_node.name))),call_module_node.name+format],{"fillcolor":color,"shape":"circle"}))
        self.edges[call_module_node.name] = ([str(id(call_module_node.name)),""],{"arrowhead":"open"})
        for i, argNode in enumerate(call_module_node.args):
            if isinstance(argNode,Node):
                # argNode is input
                if argNode.name == self.input_name:
                    self.edges[self.input_name+target_name] = ([str(id("start")),str(id(target_name)),"input: "],{"fillcolor":"green","arrowsize":"2"})
                    name = argNode.name+target_name
                # argNode is input to multiple nodes
                elif len(self.edges[argNode.name][0])>2:
                    self.edges[argNode.name+target_name] = ([str(id(argNode.name)),str(id(target_name)),""],{"arrowhead":"open"})
                    name = argNode.name+target_name
                else:
                    self.edges[argNode.name][0].insert(1,str(id(target_name)))
                    name = argNode.name
                if argNode.name in self.sizes:
                    self.edges[name][0][-1]+=str(self.sizes[argNode.name])
            else:
                nodeName = repr(argNode)+target_name
                self.nodes.append(([str(id(nodeName))+str(i),repr(argNode)],{"fillcolor":"#D9E3DA","shape":"oval"}))
                self.edges[nodeName+str(i)] = ([str(id(nodeName))+str(i),str(id(target_name)),""],{"arrowhead":"open"})
        for kwargNode in call_module_node.kwargs.values():
            if isinstance(kwargNode,Node):
                # kwargNode is input to multiple nodes
                if len(self.edges[kwargNode.name][0])>2:
                    self.edges[kwargNode.name+target_name] = ([str(id(kwargNode.name)),str(id(target_name)),""],{"arrowhead":"open"})
                    name = kwargNode.name+target_name
                else:
                    self.edges[kwargNode.name][0].insert(1,str(id(target_name)))
                    name = kwargNode.name
                if kwargNode.name in self.sizes:
                    self.edges[name][0][-1]+=str(self.sizes[kwargNode.name])
        return super().call_module(target,args,kwargs)

    def output(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        # Blue arrow with output text
        output_node = self.nodeDict[target]
        for argNode in output_node.args:
            self.edges[argNode.name][0].insert(1,str(id("end")))
            self.edges[argNode.name][1]['arrowhead'] = "normal"
            self.edges[argNode.name][1]['arrowsize'] = "2"
            self.edges[argNode.name][1]['fillcolor'] = "blue"
            self.edges[argNode.name][0][2] = "output: "+str(self.sizes[argNode.name])
        for nargs,nkwargs in self.nodes:
            self.pygraph.node(*nargs,**nkwargs)
        for eargs,ekwargs in self.edges.values():
            if len(eargs)==3:
                self.pygraph.edge(*eargs,**ekwargs)
        return super().output(target,args,kwargs)

def visualize_graph(model : torch.nn.Module, input : torch.Tensor, file_name = 'graph', tracer = QuantTracer()) -> Any:
    """ Saves the graph to file_name if given and returns the pygraph object
        Example use cases:
        visualizing non fx transformed:
            visualize_graph(torch.nn.Linear(64,64),torch.rand(1,64)) 
        visualizing fx transformed:
            net = torch.nn.Linear(64,64)
            gm = cast_input_output_transform(net)
            visualize_graph(gm,torch.rand(1,64))
    """
    if not isinstance(model,fx.GraphModule):
        graph = tracer.trace(model)
        model = fx.GraphModule(tracer.root,graph)
    nodeDict = NodeDictTransformer(model).transform()
    gi = GraphvizInterpreter(model,nodeDict)
    gi.run(input)
    gi.pygraph.render(filename = file_name)

