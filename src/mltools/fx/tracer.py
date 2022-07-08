#!/usr/bin/env python3
#
import torch
from torch import fx
from .. import numerical
from .. import sparse


class QuantTracer(fx.Tracer):
    def __init__(self) -> None:
        super().__init__()

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        is_leaf = isinstance(
            m,
            (
                numerical.CastTo,
                sparse.Sparsify,
            ),
        )
        return (
            (
                is_leaf
                or m.__module__.startswith("torch.nn")
                or m.__module__.startswith("corsair.nn")
            )
            and not isinstance(m, torch.nn.Sequential)
            or super().is_leaf_module(m, module_qualified_name)
        )
