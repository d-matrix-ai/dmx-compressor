#!/usr/bin/env python3

import torch

from torch.export import ExportGraphSignature, ExportedProgram


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


def get_workload_graphmodule(wl):
    W = wl()
    x = W.create_batch(1)
    W.model(x["input_ids"])
    m = W.model.body._gm
    return m.to('cpu'), x["input_ids"].to('cpu')


def test_distilgpt2():
    """
    Example:
        >>> result = test_distilgpt2(); print(result.graph.print_tabular())
    """

    from mlreferences import distilgpt2 as wl

    m, x = get_workload_graphmodule(wl)
    qdqm = qDq_transform(m)
    prog = torch.export.export(qdqm, (x,))
    import dmir_compiler
    from dmir_compiler import DMIRCompilerConfigs

    config = DMIRCompilerConfigs["stablehlo-dmir"]()
    config.use_fx_importer = True
    config.use_tracing = True
    config.use_sharding = False
    config.generate_artifacts = True
    config.decomposition_ops = [torch.ops.aten.split.Tensor, torch.ops.aten.split_with_sizes, torch.ops.aten.t]

    module = dmir_compiler.compile(
        qdqm, x, "distilgpt2", config
    )

    return module


if __name__ == "__main__":
    module = test_distilgpt2()
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))
