#!/usr/bin/env python3

import torch.fx as fx
from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate
from torch.fx.proxy import Proxy
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from mltools.numerical import CastTo
from mltools.utils import load_config_file
from mltools.sparse import Sparsify
from mltools.approximate import Approximator
from mltools.corsair import CorsairConfig



class InputOutputTransformer(fx.Transformer):
    def __init__(self,module:fx.GraphModule,scopeDict:dict = None,cfg = None):
        super().__init__(module)
        self.scopeDict = scopeDict
        self.config=None
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
        cast_name = target+"_cast"
        cast_format = "SAME"
        # Find input_cast format in cfg if exists
        if self.config:
            layer_key = layer.split('__')[-1]
            if layer_key:
                cast_format = self.config[layer_key]['input_format']
                
        self.module.add_submodule(cast_name,CastTo(format=cast_format))
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
        cast_name = target+"_cast"
        cast_format = "SAME"
        # Find output_cast format in cfg if exists
        if self.config:
            layer_key = layer.split('__')[-1]
            if layer_key:
                cast_format = self.config[layer_key]['output_format']
                
        self.module.add_submodule(cast_name,CastTo(format=cast_format))
        self.new_graph.inserting_before(output_node)
        output_node_cast = self.new_graph.create_node(
            "call_module", cast_name, args=(output_node.prev,)
        )
       
        self.new_graph.erase_node(output_node)
        return Proxy(output_node_cast, self.tracer)

    def get_attr(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Proxy:
        get_attr_node = self.new_graph.get_attr(target)    
        prev_node = get_attr_node
        # Default cast
        layer = self.scopeDict[get_attr_node.name][0]
        cast_name = target+"_cast"
        cast_format = "SAME"
        # Default sparsifier
        sparsify_format = "DENSE"
        sparsify_name = target+"_sparsifier"
        layer_key = layer.split('__')[-1]
        # Find casting and sparsifier format in config if exists
        if self.config:
            if layer_key:
                if "weight" in target:
                    cast_format = self.config[layer_key]['weight_format']
                    #Add sparsifier
                    sparsify_format = self.config[layer_key]['weight_sparseness']
                               
                else:
                    cast_format = self.config[layer_key]['bias_format']

        self.module.add_submodule(cast_name,CastTo(format=cast_format))

        # Add sparsifier and approximator for weight nodes if needed
        if "weight" in target:
            # Sparsifier submodules needed to be added separately even for default as tensor size 
            # is not the same for every layer
            tensor_size = self.module.get_submodule(layer.split('__')[-1]).weight.size()
            self.module.add_submodule(sparsify_name,Sparsify(tensor_size,sparseness=sparsify_format))
            self.module.add_submodule(sparsify_name,Sparsify(tensor_size,sparseness="DENSE"))
            prev_node = self.new_graph.create_node(
            "call_module", sparsify_name, args=(prev_node,)
            )
            
            if self.submodules.get('approximator'):
                prev_node = self.new_graph.create_node(
                    "call_module", "approximator", args=(prev_node,)
                )
        
        get_attr_node_cast = self.new_graph.create_node(
            "call_module", cast_name, args=(prev_node,)
        )
        return Proxy(get_attr_node_cast, self.tracer)

    def call_function(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        assert callable(target)

        call_fnc_node = self.new_graph.call_function(target)
        # Observed that inputs to the functions will be wrapped in proxies, parameters of 
        # functions is not wrapped in proxies. We need to do a unwrap for proxies before passing to new node. 
        new_kwargs=dict()
        for k in kwargs.keys():
            if isinstance(kwargs[k],Proxy):
                new_kwargs[k] = kwargs[k].node
            else:
                new_kwargs[k] = kwargs[k]
        new_args = ()
        for arg in args:
            if isinstance(arg,Proxy):
                new_args+=(arg.node,)
            else:
                new_args+=(arg,)
        call_fnc_node.args = new_args
        call_fnc_node.kwargs = new_kwargs
        approx_name = call_fnc_node.name+"_approx"
        self.module.add_submodule(approx_name,Approximator())
        call_fnc_node_approx = self.new_graph.create_node(
            "call_module", approx_name, args=(call_fnc_node,)
        )
        return Proxy(call_fnc_node_approx, self.tracer)


        
