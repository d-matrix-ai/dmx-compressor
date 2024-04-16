#!/usr/bin/env python3

import torch

from torch.export import ExportGraphSignature, ExportedProgram
from torch.export.graph_signature import InputSpec, InputKind

from torch_mlir.extras.fx_importer import FxImporter
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d

from torch.fx import replace_pattern, Transformer

from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate
from torch.fx.proxy import Proxy
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import functools
import itertools
import inspect

import mltools.numerical.custom_lib
from mltools.fx.transform import substitute_transform, qDq_transform

from torch.fx.experimental.proxy_tensor import make_fx, get_isolated_graphmodule

def get_relu_replacement():
    """
    Demo to show replacement
    """
    def pattern(x):
        return torch.ops.aten.relu.default(x)
    def replacement(x):
        return torch.ops.dmx.custom_relu.default(torch.ops.aten.relu.default(x))

    return pattern, replacement

def wrap_relu(prog: ExportedProgram):
    """ Wraps a relu in dmx custom op """
    pat, rep = get_relu_replacement()
    replace_pattern(prog.graph_module, pat, rep)
    return True

def replace_all_named_parameters(prog: ExportedProgram):
    dict = prog.graph_signature.inputs_to_parameters
    class AttrDecorator(Transformer):
        def placeholder(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Proxy:
            """
            """
            assert isinstance(target, str)
            default_value = next(iter(args)) if args else inspect.Signature.empty
            if target in dict:
                print(target)
            return Proxy(self.new_graph.placeholder(target, default_value=default_value), self.tracer)
    AttrDecorator(prog.graph_module).transform()
    return True

def filter_parameters(gs: [InputSpec]):
    return filter(lambda g: g.kind == InputKind.PARAMETER, gs)


def map_gs_to_module(gs: ExportGraphSignature):
    """
    Loop over prog.graph signature, and select all PARAMETER.

    Find placeholder in graph with same name, and wrap those nodes
    """
    gs = filter_parameters(gs)
    return None



def get_workload_graphmodule(wl):
    W = wl()
    x = W.create_batch(1)
    W.model(x)
    m = W.model.body._gm
    return m, x


def test_distilgpt2():
    """
    Example:
        >>> result = test_distilgpt2(); print(result.graph.print_tabular())
    """

    from mlreferences import distilgpt2 as wl

    m, x = get_workload_graphmodule(wl)
    qdqm = qDq_transform(m)
    prog = torch.export.export(qdqm, (x["input_ids"],))

    return(prog)

if __name__ == '__main__':
    import xdoctest
    xdoctest.doctest_module(__file__)
