from torch.fx import Proxy
from torch import nn, fx
import torch
from dmx.compressor.fx.transformer.utils import (
    process_args,
    process_kwargs,
    dmx_aware_mapping,
    dmx_aware_function_mapping_export,
    dmxnn,
)
import inspect
from .utils import get_name_for_func_nodes
import transformers


def get_nested_attr(obj, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


class ExportSubstituteTransformer(torch.fx.Transformer):
    def __init__(
        self,
        gm: fx.GraphModule,
        model: nn.Module,
        scope: str = "",
        node_inputs: dict = None,
    ):
        super().__init__(gm)
        self.model = model
        self.module = gm
        self.scope = scope
        self.old_nodes = [n for n in gm.graph.nodes]
        self.node_counter = 0
        self.node_inputs = node_inputs
        self.dmx_aware_function_mapping_export = (
            dmx_aware_function_mapping_export.copy()
        )

    def placeholder(self, target, args, kwargs):
        self.node_counter += 1
        return super().placeholder(target, args, kwargs)

    def add_submod(self, name, mod):
        """
        this function will try to reuse modules in old_gm if possible
        """
        if hasattr(self.model, "_gms"):
            gms = self.model._gms.values()
        else:
            gms = (self.model._gm,)
        for old_gm in gms:
            try:
                self.module.add_submodule(name, old_gm.get_submodule(self.scope + name))
                return
            except:
                continue
        self.module.add_submodule(name, mod)

    def call_module(self, target, args, kwargs):
        self.node_counter += 1
        try:
            orig_mod = get_nested_attr(self.model, self.scope + target)
        except:
            # in case the module is not found, it means it is a transformed module
            orig_mod = self.module.get_submodule(target)
        mod_type = node_key = type(orig_mod).__module__ + "." + type(orig_mod).__name__
        if mod_type in dmx_aware_mapping:
            self.add_submod(target, dmx_aware_mapping[node_key].from_raw(orig_mod))
            ## special treatment for modules not packed by unflatten (appearing as InterpreterModules), where the input is different from the original module (usually with additional inputs for shapes)
            if isinstance(orig_mod, transformers.pytorch_utils.Conv1D):
                new_node = self.new_graph.create_node(
                    "call_module",
                    target,
                    args=(args[-1].node,),
                )
            else:
                new_node = self.new_graph.create_node(
                    "call_module", target, args=tuple(arg.node for arg in args)
                )
                # Special treatment for RMSNorm, Unflattened RMSNorm output sometimes is a tuple of two elements, but original RMSNorm output is a tensor
                if isinstance(self.module.get_submodule(target), dmxnn.RMSNorm):
                    if (
                        str(self.old_nodes[self.node_counter].target)
                        == "<built-in function getitem>"
                        and str(self.old_nodes[self.node_counter + 1].target)
                        == "<built-in function getitem>"
                    ):
                        new_node = self.new_graph.call_function(
                            tuple, args=([args[0].node, new_node],)
                        )

            return Proxy(new_node, self.tracer)

        # module is a transformed module
        elif isinstance(orig_mod, (dmxnn.DmxModule, fx.GraphModule)):
            return super().call_module(target, args, kwargs)
        else:
            curr_mod = self.module.get_submodule(target)
            transformer = ExportSubstituteTransformer(
                fx.GraphModule(curr_mod, curr_mod.graph),
                self.model,
                self.scope + target + ".",
                node_inputs=self.node_inputs,
            )
            transformed = transformer.transform()
            transformed.graph.eliminate_dead_code()
            # we do not want to reuse this submod since it is traced through and control flows might changed
            self.module.add_submodule(target, transformed)
            new_node = self.new_graph.create_node(
                "call_module", target, args=tuple(arg.node for arg in args)
            )
            return Proxy(new_node, self.tracer)

    def call_method(self, target, args, kwargs):
        self.node_counter += 1
        return super().call_method(target, args, kwargs)

    def call_function(self, target, args, kwargs):
        node = self.old_nodes[self.node_counter]
        name = node.name
        self.node_counter += 1
        if str(target) in self.dmx_aware_function_mapping_export:
            empty_mod = self.dmx_aware_function_mapping_export[str(target)]()
            if isinstance(empty_mod, dmxnn.ResAdd):
                mod_args, mod_kwargs = self.node_inputs[name]
                # only substitute add with both inputs as tensors
                if not isinstance(mod_args[0], torch.Tensor) or not isinstance(
                    mod_args[1], torch.Tensor
                ):
                    return super().call_function(target, args, kwargs)
                name = "resadd"
            elif isinstance(empty_mod, dmxnn.ActActMatMul):
                name = "actmatmul"
            elif isinstance(empty_mod, dmxnn.Mul):
                mod_args, mod_kwargs = self.node_inputs[name]
                # only substitute add with both inputs as tensors
                if not isinstance(mod_args[0], torch.Tensor) or not isinstance(
                    mod_args[1], torch.Tensor
                ):
                    return super().call_function(target, args, kwargs)
                name = "mul"
            new_name = get_name_for_func_nodes(
                name,
                self.new_graph._graph_namespace._used_names,
                self.new_graph._graph_namespace._base_count,
            )
            accepted_kwarg_keys = inspect.signature(
                empty_mod.__init__
            ).parameters.keys()
            initkwargs = {}
            newkwargs = {}
            for key, value in kwargs.items():
                if key in accepted_kwarg_keys:
                    initkwargs[key] = value
                else:
                    newkwargs[key] = value
            newmod = self.dmx_aware_function_mapping_export[str(target)](**initkwargs)
            newargs, newkwargs = process_args(args), process_kwargs(newkwargs)

            if self.dmx_aware_function_mapping_export[
                str(target)
            ].is_compound:  # if the module is a compound op
                mod_args, mod_kwargs = self.node_inputs[name]
                # remove kwargs used for init
                mod_kwargs = {
                    k: v for k, v in mod_kwargs.items() if k not in initkwargs
                }
                module_graph = newmod.module_graph(*mod_args, **mod_kwargs)
                # last node inserted from merging module_graph
                new_node = self.merge_graph(module_graph, new_name, newargs, newkwargs)
            else:
                self.add_submod(new_name, newmod)
                new_node = self.new_graph.create_node(
                    "call_module",
                    new_name,
                )
                new_node.args = newargs
                new_node.kwargs = newkwargs
            return Proxy(new_node, self.tracer)

        return super().call_function(target, args, kwargs)

    def get_attr(self, target, args, kwargs):
        self.node_counter += 1
        return super().get_attr(target, args, kwargs)

    def merge_graph(self, subgraph, name, args, kwargs):
        """
        This function is to merge subgraph to graph
        """
        # add submodules of subgraph to self
        for n, m in subgraph.named_children():
            self.add_submod(name + "." + n, m)

        # add nodes of subgraph to self.new_graph
        node_mapping = {}
        arg_counter = 0
        for node in subgraph.graph.nodes:
            newargs = tuple(
                node_mapping[arg.name] if isinstance(arg, fx.Node) else arg
                for arg in node.args
            )
            newkwargs = {
                k: node_mapping[str(v)] if isinstance(v, fx.Node) else v
                for k, v in node.kwargs.items()
            }
            # placeholder nodes should be skipped and they should be mapped to args/kwargs input to the subgraph
            if node.op == "placeholder":
                if arg_counter < len(args):
                    node_mapping[node.name] = args[arg_counter]
                    arg_counter += 1
                elif node.name in kwargs.keys():
                    node_mapping[node.name] = kwargs[node.name]
                else:
                    raise ValueError("Input to the compound function is incorrect!")

            elif node.op == "call_function" or node.op == "call_method":
                new_node = self.new_graph.create_node(
                    node.op,
                    node.target,
                    newargs,
                    newkwargs,
                    name=name + "." + node.name,
                )
                node_mapping[node.name] = new_node

            elif node.op == "call_module" or node.op == "get_attr":
                new_node = self.new_graph.create_node(
                    node.op,
                    name + "." + node.target,
                    newargs,
                    newkwargs,
                )
                node_mapping[node.name] = new_node
        return new_node
