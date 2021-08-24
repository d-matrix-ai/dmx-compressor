from os import name
from parse import parse
from torch.fx import graph
from utils.dmir_pb2 import *
from types import CodeType, FunctionType, ModuleType
from typing import (
    Any,
    Dict,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    List,
    Callable,
    Union,
)
import torch
import torch.fx as fx
import numerical, sparse

# # identifier strings of operators, compatible with ONNX naming
# CASTTO = "CastTo"
# ADD = "Add"
# MUL = "Mul"
# TRANSPOSE = "Transpose"
# MATMUL = "MatMul"
# CONV2D = "Conv"
# AVGPOOL2D = "AveragePool"
# MAXPOOL2D = "MaxPool"
# BATCHNORM2D = "BatchNormalization"
# LAYERNORM = "LayerNormalization"
# DROPOUT = "Dropout"
# SOFTMAX = "Softmax"
# RELU = "Relu"
# RELU6 = "Relu6"
# TANH = "Tanh"


class DMIRTracer(fx.Tracer):
    r"""
    This is a DMIR-0 tracer that takes a PyTorch module and generates python code that constructs DMIR-0 of it.
    """

    def __init__(self, flat: bool = False) -> None:
        super().__init__()
        self.flat = flat

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        is_leaf = isinstance(
            m,
            (
                numerical.CastTo,
                sparse.Sparsify,
            ),
        )
        if self.flat:
            return is_leaf
        else:
            return (
                is_leaf
                or m.__module__.startswith("torch.nn")
                or m.__module__.startswith("corsair.nn")
                and not isinstance(m, torch.nn.Sequential)
            )

    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        module_qualified_name = self.path_of_module(m)
        return (
            self.create_proxy("call_module", module_qualified_name, args, kwargs)
            if self.is_leaf_module(m, module_qualified_name)
            else forward(*args, **kwargs)
        )


def _torch_qualified_name(name: str) -> str:
    segs = name.split(".")
    name = ".".join(f"[{seg}]" if seg.isnumeric() else seg for seg in segs)
    return name.replace(".[", "[")


def _make_var_name(name: str) -> str:
    return (name + "_").replace(".", "__")


def dump(
    m: torch.nn.Module,
    name: str = "model",
    flat: bool = False,
    omit_value: bool = False,
) -> Graph:
    # TODO: treat numerical constant args as input
    # TODO: add attributes of modules
    # TODO: map built-in functions to ONNX ops

    tracer = DMIRTracer(flat=flat)
    traced = tracer.trace(m)

    input = []
    output = []
    intermediate = []
    subgraph = []
    dependency = []

    for node in traced.nodes:
        if node.op == "placeholder":
            input.append(
                Tensor(
                    name=_make_var_name(node.name),
                    format=FLOAT,
                )
            )
        elif node.op == "get_attr":
            _p = eval(_torch_qualified_name(f"m.{node.target}"))
            input.append(
                Tensor(
                    name=_make_var_name(node.name),
                    shape=_p.shape,
                    format=FLOAT,
                    value=[] if omit_value else _p.data.view(-1).numpy().tolist(),
                )
            )
        elif node.op == "output":
            output.append(
                Tensor(
                    name=_make_var_name(node.name),
                    format=FLOAT,
                )
            )
        elif node.op in ("call_function", "call_method", "call_module"):
            intermediate.append(
                Tensor(
                    name=_make_var_name(node.name),
                    format=FLOAT,
                )
            )
            dependency.append(
                Dependency(
                    operation=traced._target_to_str(node.target),
                    argument=(_make_var_name(n.__str__()) for n in node.args),
                    result=(_make_var_name(node.name),),
                )
            )
            if node.op == "call_module":
                _m = eval(_torch_qualified_name(f"m.{node.target}"))
                if isinstance(_m, numerical.CastTo):
                    subgraph.append(
                        Graph(
                            name=node.name,
                            op_type="cast_to",
                        )
                    )
                elif isinstance(_m, sparse.Sparsify):
                    subgraph.append(
                        Graph(
                            name=node.name,
                            op_type="sparsify",
                        )
                    )
                    # add elem-wise mul here
                else:
                    subgraph.append(
                        dump(_m, name=node.name, flat=flat, omit_value=omit_value)
                    )
            else:  # built-in function or tensor method
                subgraph.append(
                    Graph(
                        name=node.name,
                        op_type=traced._target_to_str(node.target),
                    )
                )
        else:
            raise RuntimeError(f"illegal FXIR node opcode {node.op}")

    return Graph(
        name=name,
        input=input,
        output=output,
        intermediate=intermediate,
        dependency=dependency,
        subgraph=subgraph,
    )


if __name__ == "__main__":
    import torch.nn as nn
    from models import LeNet

    import corsair

    # class MyModule(torch.nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.param = torch.nn.Parameter(torch.rand(3, 4))
    #         self.linear = torch.nn.Linear(4, 5)

    #     def forward(self, x):
    #         return torch.topk(
    #             torch.sum(self.linear(x + self.linear.weight).relu(), dim=-1), 3
    #         )

    model = nn.Sequential(
        # nn.CastTo(format="SAME"),
        nn.CastTo(format="BFP[4|8]{64,-1}(N)"),
        LeNet([512, 512]),
    )
    # model = MyModule()

    g = dump(model, name="lenet-512-512", flat=False, omit_value=True)
    breakpoint()
