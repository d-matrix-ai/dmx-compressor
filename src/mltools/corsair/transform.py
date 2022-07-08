from ast import Str
from logging import PlaceHolder
from modulefinder import Module
import sys
import yaml
import re
import torch
from types import SimpleNamespace
from .nn import *
from mltools import corsair, dmir, numerical
from mltools.utils import load_config_file, graph_utils, save_config_file, print_model_tree
from sol.src.sys.corsair_hw import *
from sol.src.sol_sim import *
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import torch.nn as nn
from ..numerical import CastTo
from ..fx import QuantTracer, InputOutputTransformer
from torch.fx import GraphModule


def aware():
    # add new torch.nn modules for corsair
    torch.nn.CastTo = CastTo
    torch.nn.Sparsify = Sparsify
    torch.nn.Approximate = Approximate
    # overload existing torch.nn modules for corsair
    torch.nn.Linear = Linear
    torch.nn.Conv2d = Conv2d
    torch.nn.AdaptiveAvgPool2d = AdaptiveAvgPool2d
    torch.nn.MaxPool2d = MaxPool2d
    torch.nn.BatchNorm2d = BatchNorm2d
    torch.nn.LayerNorm = LayerNorm
    torch.nn.Dropout = Dropout
    torch.nn.Softmax = Softmax
    torch.nn.ReLU = ReLU
    torch.nn.ReLU6 = ReLU6
    torch.nn.Tanh = Tanh
    torch.nn.GELU = GELU


def cast_input_output_transform(
    root: torch.nn.Module,
    input_fn: nn.Module = CastTo(),
    output_fn: nn.Module = CastTo(),
    weight_fn: nn.Module = CastTo(),
    concrete_args: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    tracer = QuantTracer()
    graph = tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    gm = GraphModule(tracer.root, graph, name)

    # TODO assertions
    assert gm.add_submodule("input_cast", input_fn), True
    assert gm.add_submodule("output_cast", output_fn), True
    assert gm.add_submodule("weight_cast", weight_fn), True
    transformed = InputOutputTransformer(gm).transform()

    # TODO These lines not tested
    # for i in nodeList:
    #     if i.op == "call_module":
    #         transformed_gm = cast_input_output_transform(gm.get_submodule(i.target))
    #         transformed.delete_submodule(i.target), True
    #         transformed.add_submodule(i.target, transformed_gm), True

    return transformed


class Model(torch.nn.Module):
    r"""
    Container for a DNN model to be deployed
    - body to be mapped on device
    - head and tail to be executed on host, corresponding to pre- and post-processing
    - equipped with corsair-aware transformation
    """

    def __init__(
        self, body, head=torch.nn.Identity(), tail=torch.nn.Identity()
    ) -> None:
        super().__init__()
        self.body = body
        self.head = head
        self.tail = tail

    def forward(self, input, dmir_executor=None):
        # NOTE: only a single input is allowed
        output = self.head(input)
        if isinstance(output, torch.Tensor):
            output = (output,)
        output = self.body(*output) if dmir_executor is None else dmir_executor(*output)
        output = self.tail(output)
        return output

    def transform(self, config="configs/corsair.yaml", *transformations):
        r"""
        Transform with Corsair-specific numerics/sparsity/logics
        NOTE: only staticly declared CorsairModule(s) are to be transformed
        """
        if isinstance(config, str):
            config = CorsairConfig.from_yaml(config)

        for n, m in self.named_corsair_modules():
            m._transform(config[n])

        for tr in transformations:
            tr.apply_to(self)

        return self

    @property
    def corsair_config(self):
        return CorsairConfig.from_model(self)

    @property
    def corsair_module_names(self):
        return self.corsair_config.module_names

    def named_corsair_modules(self):
        return ((n, m) for n, m in self.body.named_modules() if is_configurable(m))

    def freeze(self, config_file="./config.yaml"):
        CorsairConfig.from_model(self, freeze=True).to_yaml(config_file)

    def thaw(self, config_file="./config.yaml"):
        self.transform(config_file)

    def print_model_tree(self, include_type=False):
        print_model_tree(self, include_type)

    def dmir_graph(self, subgraph: str, sample_input, **kwargs):
        return dmir.dump(
            eval(f"self.body.{subgraph}" if subgraph else "self.body"),
            sample_input,
            **kwargs,
        )

    def fx_graph(self, sample_input, **kwargs):
        return dmir.parse_fx(
            self.body,
            self.head(sample_input),
            graph_dict=dict(),
            **kwargs,
        )

    def sol_analyze(self, sample_input, corsair_hw=Slice(), **kwargs):
        def filter_sol_output(perf_data, power_data):
            perf_data = perf_data["SOL_Performance_Analysis"]
            power_data = power_data["On-Chip_Dynamic_Power"]

            # remove derived utilization percentages from power_data:
            for k in power_data:
                power_data[k] = power_data[k]["power(mW)"]

            return perf_data, power_data

        graph = self.dmir_graph(sample_input)

        perf_data, power_data = analyze(graph, corsair_hw=corsair_hw, **kwargs)
        perf_data, power_data = filter_sol_output(perf_data, power_data)

        return perf_data, power_data


class CorsairConfig(dict):
    r"""
    This is a dict of Corsair-specific configurations for a corsair.Model
    This defines the 'states' to be optimized
    """

    @classmethod
    def from_model(cls, model: Model, freeze=False):
        return cls(
            {n: m._corsair_config(freeze) for n, m in model.named_corsair_modules()}
        )

    @classmethod
    def from_yaml(cls, fname):
        return cls(load_config_file(fname))

    def to_yaml(self, fname):
        save_config_file(dict(self), fname)

    @property
    def module_names(self):
        return self.keys()


class CorsairTransformation(SimpleNamespace):
    r"""
    This is a rule that specifies how to transform from CorsairConfig to CorsairConfig
    This defines the 'action' in the state space
    """

    def __init__(
        self,
        module_types=(),
        name_re: Str = "",
        module_config: CorsairModuleConfig = CorsairModuleConfig(),
    ) -> None:
        assert all([issubclass(mt, CorsairModule) for mt in module_types])
        self.module_types = module_types
        self.name_rule = re.compile(name_re)
        self.module_config = module_config

    def names_in(self, model_or_config: Union[Model, CorsairConfig]):
        config = (
            model_or_config
            if isinstance(model_or_config, CorsairConfig)
            else CorsairConfig.from_model(model_or_config, freeze=True)
        )
        return [
            n
            for n in config.module_names
            if getattr(corsair.nn, config[n]["instance"]) in self.module_types
            and self.name_rule.match(n)
        ]

    def apply_to(self, model_or_config: Union[Model, CorsairConfig]):
        target_module_names = self.names_in(model_or_config)
        if isinstance(model_or_config, Model):
            for n, m in model_or_config.named_corsair_modules():
                if n in target_module_names and type(m) in self.module_types:
                    m._transform(self.module_config)
        else:
            config = model_or_config
            for n in (
                target_module_names
                and getattr(corsair.nn, config[n]["instance"]) in self.module_types
            ):
                config[n].update(self.module_config)
