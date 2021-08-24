from os import name
from parse import parse
from torch.fx import graph
from utils.dmir_pb2 import *
from google.protobuf.json_format import MessageToJson
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
import numerical, sparse, approximate


FORMAT_DICT = {
    torch.float32: FLOAT32,
    torch.float: FLOAT,
    torch.float64: FLOAT64,
    torch.double: DOUBLE,
    torch.float16: FLOAT16,
    torch.half: HALF,
    torch.bfloat16: BFLOAT16,
    torch.uint8: UINT8,
    torch.int8: INT8,
    torch.int16: INT16,
    torch.short: SHORT,
    torch.int32: INT32,
    torch.int: INT,
    torch.int64: INT64,
    torch.long: LONG,
    torch.bool: BOOL,
}


class TensorMetadata(NamedTuple):
    r"""
    TensorMetadata is a structure containing pertinent information
    about a tensor within a PyTorch program.
    Taken from https://github.com/pytorch/pytorch/blob/master/torch/fx/passes/shape_prop.py
    """
    shape: torch.Size
    dtype: torch.dtype
    requires_grad: bool
    stride: Tuple[int]
    memory_format: Optional[torch.memory_format]

    # Quantization metadata
    is_quantized: bool
    qscheme: Optional[torch.qscheme]
    q_scale: Optional[float]
    q_zero_point: Optional[int]


def extract_tensor_metadata(result: torch.Tensor) -> TensorMetadata:
    r"""
    Extract a TensorMetadata NamedTuple describing `result`.
    Taken from https://github.com/pytorch/pytorch/blob/master/torch/fx/passes/shape_prop.py
    """
    shape = result.shape
    dtype = result.dtype
    requires_grad = result.requires_grad
    stride = result.stride()

    memory_formats = {
        torch.contiguous_format,
        torch.channels_last,
        torch.channels_last_3d,
    }

    memory_format = None

    for query_format in memory_formats:
        if result.is_contiguous(memory_format=query_format):
            memory_format = query_format
            break

    is_quantized = result.is_quantized
    qscheme = None
    q_scale = None
    q_zero_point = None

    if is_quantized:
        qscheme = result.qscheme()

        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            q_scale = result.q_scale()
            q_zero_point = result.q_zero_point()

    return TensorMetadata(
        shape,
        dtype,
        requires_grad,
        stride,
        memory_format,
        is_quantized,
        qscheme,
        q_scale,
        q_zero_point,
    )


class ShapeProp(fx.Interpreter):
    r"""
    Taken from https://github.com/pytorch/pytorch/blob/master/torch/fx/passes/shape_prop.py
    """

    def run_node(self, n: fx.node.Node) -> Any:
        result = super().run_node(n)

        found_tensor = False

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return extract_tensor_metadata(obj)
            else:
                return obj

        meta = fx.node.map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta["tensor_meta"] = meta

        n.meta["type"] = type(result)
        return result

    def propagate(self, *args):
        return super().run(*args)


class DMIRTracer(fx.Tracer):
    r"""
    This is a DMIR-0 tracer that takes a PyTorch module and generates DMIR-0 of it.
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
    assert not name.isnumeric(), "numerical args not support as of now"
    # TODO: treat numerical constant args as an input node
    return (name + "_").replace(".", "__")


def _legal_op_type(opname: str) -> str:
    # TODO: map built-in functions to ONNX style ops
    return opname


def _legal_format(torch_format):
    return (
        FORMAT_DICT[torch_format] if torch_format in FORMAT_DICT.keys() else UNDEFINED
    )


def _nn_module_meta(m: torch.nn.Module) -> str:
    # TODO: add metadata of modules that is necessary and useful for transformations
    return str(m)


def _corsair_specific_attributes(m: torch.nn.Module):
    attr = []
    if hasattr(m, "format"):
        attr.append(
            Attribute(
                kind=Attribute.STRING,
                name="numerical_format",
                string_value=repr(m.format),
            )
        )
    if hasattr(m, "sparseness"):
        attr.append(
            Attribute(
                kind=Attribute.STRING,
                name="sparseness_pattern",
                string_value=repr(m.sparseness),
            )
        )
    if hasattr(m, "approximator"):
        attr.append(
            Attribute(
                kind=Attribute.STRING,
                name="approximation_function",
                string_value=repr(m.approximator.function),
            )
        )
    return attr


def dump(
    m: torch.nn.Module,
    *sample_input: torch.Tensor,
    name: str = "model",
    flat: bool = False,
    omit_value: bool = False,
    metadata: str = "",
) -> Graph:

    tracer = DMIRTracer(flat=flat)
    gm = fx.GraphModule(root=m, graph=tracer.trace(m))
    ShapeProp(gm).propagate(*sample_input)
    traced = gm.graph

    input = []
    output = []
    intermediate = []
    subgraph = []
    dependency = []

    for node in traced.nodes:
        if node.op == "placeholder":  # dynamic inputs
            input.append(
                Tensor(
                    name=_make_var_name(node.name),
                    shape=node.meta["tensor_meta"].shape,
                    format=_legal_format(node.meta["tensor_meta"].dtype),
                )
            )
        elif node.op == "get_attr":  # static inputs
            _p = eval(_torch_qualified_name(f"m.{node.target}"))
            input.append(
                Tensor(
                    name=_make_var_name(node.name),
                    shape=node.meta["tensor_meta"].shape,
                    format=_legal_format(node.meta["tensor_meta"].dtype),
                    value=[] if omit_value else _p.data.view(-1).numpy().tolist(),
                )
            )
        elif node.op == "output":  # output
            output.append(
                Tensor(
                    name=_make_var_name(node.name),
                    shape=node.meta["tensor_meta"].shape,
                    format=_legal_format(node.meta["tensor_meta"].dtype),
                )
            )
        elif node.op in ("call_function", "call_method", "call_module"):  # subgraphs
            intermediate.append(
                Tensor(
                    name=_make_var_name(node.name),
                    shape=node.meta["tensor_meta"].shape,
                    format=_legal_format(node.meta["tensor_meta"].dtype),
                )
            )
            if node.op == "call_module":
                _m = eval(_torch_qualified_name(f"m.{node.target}"))
                dependency.append(
                    Dependency(
                        operation=node.name,
                        argument=(_make_var_name(n.__str__()) for n in node.args),
                        result=(_make_var_name(node.name),),
                        attribute=_corsair_specific_attributes(_m),
                    )
                )
                if isinstance(_m, numerical.CastTo):
                    subgraph.append(
                        Graph(
                            name=node.name,
                            op_type=_legal_op_type("cast_to"),
                            metadata=_nn_module_meta(_m),
                        )
                    )
                elif isinstance(_m, sparse.Sparsify):
                    subgraph.append(
                        Graph(
                            name=node.name,
                            input=(
                                Tensor(
                                    name="dense",
                                    shape=node.args[0].meta["tensor_meta"].shape,
                                    format=_legal_format(
                                        node.args[0].meta["tensor_meta"].dtype
                                    ),  # this is a dynamic input
                                ),
                                Tensor(
                                    name="mask",
                                    shape=node.meta["tensor_meta"].shape,
                                    format=_legal_format(
                                        node.meta["tensor_meta"].dtype
                                    ),
                                    value=[]
                                    if omit_value
                                    else _m.mask.data.view(-1).numpy().tolist(),
                                ),  # this is a static input
                            ),
                            output=(
                                Tensor(
                                    name="sparse",
                                    shape=node.meta["tensor_meta"].shape,
                                    format=_legal_format(
                                        node.meta["tensor_meta"].dtype
                                    ),
                                ),
                            ),
                            subgraph=(
                                Graph(
                                    name="mul",
                                    op_type=_legal_op_type(
                                        traced._target_to_str(torch.mul)
                                    ),
                                ),
                            ),
                            dependency=(
                                Dependency(
                                    operation="mul",
                                    argument=("dense", "mask"),
                                    result=("sparse",),
                                ),
                            ),
                            metadata=_nn_module_meta(_m),
                        )
                    )
                else:  # custom modules
                    subgraph.append(
                        dump(
                            _m,
                            *[
                                torch.randn(
                                    arg.meta["tensor_meta"].shape,
                                    dtype=arg.meta["tensor_meta"].dtype,
                                )
                                for arg in node.args
                            ],
                            name=node.name,
                            flat=flat,
                            omit_value=omit_value,
                            metadata=_nn_module_meta(_m),
                        )
                    )
            else:  # built-in function or tensor method
                dependency.append(
                    Dependency(
                        operation=node.name,
                        argument=(_make_var_name(n.__str__()) for n in node.args),
                        result=(_make_var_name(node.name),),
                    )
                )
                subgraph.append(
                    Graph(
                        name=node.name,
                        op_type=_legal_op_type(traced._target_to_str(node.target)),
                        metadata=_nn_module_meta(_m),
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
        metadata=metadata,
    )


def save_to_file(model: Graph, filename: str, format="binary") -> None:
    if format == "binary":
        with open(filename, "wb") as f:
            f.write(model.SerializeToString())
    elif format == "json":
        with open(filename, "w") as f:
            f.write(MessageToJson(model))
    else:
        raise RuntimeError(f"unsupported DMIR file format: {format}")


if __name__ == "__main__":
    import torch.nn as nn
    from models import LeNet

    import corsair

    # # class MyModule(torch.nn.Module):
    # #     def __init__(self):
    # #         super().__init__()
    # #         self.param = torch.nn.Parameter(torch.rand(3, 4))
    # #         self.linear = torch.nn.Linear(4, 5)

    # #     def forward(self, x):
    # #         return torch.topk(
    # #             torch.sum(self.linear(x + self.linear.weight).relu(), dim=-1), 3
    # #         )

    # model = nn.Sequential(
    #     # nn.CastTo(format="SAME"),
    #     nn.CastTo(format="BFP[4|8]{64,-1}(N)"),
    #     LeNet([512, 512]),
    # )
    # # model = MyModule()

    # g = dump(model, name="lenet-512-512", flat=False, omit_value=True)
    # breakpoint()

    # class TwoLayerNet(torch.nn.Module):
    #     def __init__(self, D_in, H, D_out):
    #         super(TwoLayerNet, self).__init__()
    #         self.linear1 = torch.nn.Linear(D_in, H)
    #         self.linear2 = torch.nn.Linear(H, D_out)

    #     def forward(self, x):
    #         h_relu = self.linear1(x).clamp(min=0)
    #         y_pred = self.linear2(h_relu)
    #         return y_pred

    # N, D_in, H, D_out = 64, 1000, 100, 10
    # x = torch.randn(N, D_in)
    # model = TwoLayerNet(D_in, H, D_out)
    # gm = torch.fx.symbolic_trace(model)
    # sample_input = torch.randn(50, D_in)
    # ShapeProp(gm).propagate(sample_input)

    model = nn.Sequential(
        nn.CastTo(format="BFP[4|8]{64,-1}(N)"),
        LeNet([512, 512]),
    )

    g = dump(
        model, torch.randn(1, 784), name="lenet-512-512", flat=False, omit_value=True
    )
    breakpoint()
