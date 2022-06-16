import sys
import torch
from .nn import *
from mltools import dmir
from mltools.utils import load_config_file, graph_utils
from sol.src.sys.corsair_hw import *
from sol.src.sol_sim import *
import torch.fx as fx
from torch.fx.node import Argument, Node, Target, map_arg, map_aggregate
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

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

class CorsairTransform(fx.Transformer):
    # def call_module(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
    #     call_self, *rest_of_args = args
    #     submod  = self.fetch_attr(target)
    #     submod.forward = Linear(64,64).forward
    #     ipdb.set_trace()
    #     return self.tracer.call_module(submod, submod.forward, args, kwargs)
    #     #  return call_self(rest_of_args)

    def transform(self) -> fx.GraphModule:
        """
        Transform ``self.module`` and return the transformed
        ``GraphModule``.
        """
        result = super().run()
        if result is not None:
            def strip_proxy(a : Union[Argument, fx.Proxy]) -> Any:
                return a.node if isinstance(a, fx.Proxy) else a
            self.new_graph.output(map_aggregate(result, strip_proxy))
        if (isinstance(self.module.linear, torch.nn.Linear)):
            in_features = self.module.linear.in_features
            out_features = self.module.linear.out_features
            self.module.linear = Linear(in_features,out_features)
        return fx.GraphModule(self.module, self.new_graph)


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
        neck = self.head(input)
        butt = self.body(neck) if dmir_executor is None else dmir_executor(neck)
        output = self.tail(butt)
        return output

    def transform(self, config="configs/corsair.yaml"):
        r"""
        Transform with Corsair-specific numerics/sparsity/logics
        NOTE: only staticly declared CorsairModule(s) are to be transformed
        """
        if isinstance(config, str):
            config = load_config_file(config)

        for n, m in self.body.named_modules():
            if isinstance(m, CorsairModule):
                for r in config["transformation_rules"]:
                    if (
                        isinstance(m, eval(r["instance"]))
                        and all([_n in n for _n in r["name_includes"]])
                        and all([not _n in n for _n in r["name_excludes"]])
                    ):
                        m._transform(r["config"])

    def dmir_graph(self, sample_input, **kwargs):
        return dmir.dump(
            self.body,
            self.head(sample_input),
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
        graph = self.fx_graph(sample_input)
        return analyze(graph, corsair_hw=corsair_hw, **kwargs)
