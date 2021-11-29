from mltools.utils.dmir_pb2 import *
import itertools
import json
from google.protobuf.json_format import MessageToJson, Parse
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
from torch import fx
from mltools import numerical, sparse, approximate, corsair

__ALL__ = [
    "dump",
    "save_to_file",
    "load_from_file",
    "is_legal",
    "lower",
    "executor",
    "cpsim_executor",
]

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
                torch.nn.modules.batchnorm._BatchNorm,
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


def _make_var_name(name: str, prefix: str = "", suffix: str = "") -> str:
    # TODO: treat constant args as attributes
    if name.isnumeric() or name in ("None", "True", "False"):
        return name
    elif name.startswith("("):
        return name
    try:
        _ = float(name)
        return name
    except ValueError:
        if prefix != "":
            prefix = prefix + "_"
        if suffix != "":
            suffix = "_" + suffix
        name = f"{prefix}{name}{suffix}"
        return f"{name}_".replace(".", "__")


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


def _corsair_specific_attributes(m: torch.nn.Module) -> List[Attribute]:
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


def _tensor_meta_dict(meta):
    if isinstance(meta, list):
        assert len(meta) == 1
        meta = meta[0]
    return dict(
        shape=meta.shape,
        format=_legal_format(meta.dtype),
        is_quantized=meta.is_quantized,
        qscheme=str(meta.is_quantized) if meta.is_quantized else "",
        q_scale=meta.q_scale,
        q_zero_point=meta.q_zero_point,
    )


def _sparsifier_graph(m, node, input_names, output_names, omit_value=False):
    return Graph(
        name=node.name,
        input=(
            Tensor(
                name=_make_var_name(node.name, suffix="dense"),
                **_tensor_meta_dict(node.args[0].meta["tensor_meta"]),
            ),  # this is a dynamic input
            Tensor(
                name=_make_var_name(node.name, suffix="mask"),
                value=[]
                if omit_value
                else m.mask.data.contiguous().view(-1).numpy().tolist(),
                **_tensor_meta_dict(node.meta["tensor_meta"]),
            ),  # this is a static input
        ),
        output=(
            Tensor(
                name=_make_var_name(node.name, suffix="sparse"),
                **_tensor_meta_dict(node.meta["tensor_meta"]),
            ),
        ),
        dependency=(
            Dependency(
                operation=f"built-in:{_legal_op_type(node.graph._target_to_str(torch.mul))}",
                argument=(
                    _make_var_name(node.name, suffix="dense"),
                    _make_var_name(node.name, suffix="mask"),
                ),
                result=(_make_var_name(node.name, suffix="sparse"),),
            ),
            Dependency(
                operation=f"built-in:{_legal_op_type(node.graph._target_to_str(torch.nn.Identity))}",
                argument=(f"::{input_names[0]}",),
                result=(_make_var_name(node.name, suffix="dense"),),
            ),
            Dependency(
                operation=f"built-in:{_legal_op_type(node.graph._target_to_str(torch.nn.Identity))}",
                argument=(_make_var_name(node.name, suffix="sparse"),),
                result=(f"::{output_names[0]}",),
            ),
        ),
        metadata=_nn_module_meta(m),
    )


def _batch_norm_graph(m, node, input_names, output_names, omit_value=False):
    return Graph(
        name=node.name,
        input=(
            Tensor(
                name=_make_var_name(node.name, suffix="input"),
                **_tensor_meta_dict(node.args[0].meta["tensor_meta"]),
            ),  # this is a dynamic input
        ),
        intermediate=(
            Tensor(
                name=_make_var_name(node.name, suffix="running_mean"),
                value=[]
                if omit_value
                else m.running_mean.data.contiguous().view(-1).numpy().tolist(),
                shape=m.running_mean.shape,
                format=_legal_format(m.running_mean.dtype),
            ),  # this is a static input
            Tensor(
                name=_make_var_name(node.name, suffix="running_var"),
                value=[]
                if omit_value
                else m.running_var.data.contiguous().view(-1).numpy().tolist(),
                shape=m.running_var.shape,
                format=_legal_format(m.running_var.dtype),
            ),  # this is a static input
            Tensor(
                name=_make_var_name(node.name, suffix="weight"),
                value=[]
                if omit_value
                else m.weight.data.contiguous().view(-1).numpy().tolist(),
                shape=m.weight.shape,
                format=_legal_format(m.weight.dtype),
            ),  # this is a static input
            Tensor(
                name=_make_var_name(node.name, suffix="bias"),
                value=[]
                if omit_value
                else m.bias.data.contiguous().view(-1).numpy().tolist(),
                shape=m.bias.shape,
                format=_legal_format(m.bias.dtype),
            ),  # this is a static input
        ),
        output=(
            Tensor(
                name=_make_var_name(node.name, suffix="output"),
                **_tensor_meta_dict(node.meta["tensor_meta"]),
            ),
        ),
        dependency=(
            Dependency(
                operation=f"built-in:{_legal_op_type(node.graph._target_to_str(torch.batch_norm))}",
                argument=(
                    _make_var_name(node.name, suffix="input"),
                    _make_var_name(node.name, suffix="running_mean"),
                    _make_var_name(node.name, suffix="running_var"),
                    _make_var_name(node.name, suffix="weight"),
                    _make_var_name(node.name, suffix="bias"),
                ),
                result=(_make_var_name(node.name, suffix="output"),),
                attribute=(
                    Attribute(
                        kind=FLOAT,
                        name="momentum",
                        float_value=m.momentum,
                    ),
                    Attribute(
                        kind=FLOAT,
                        name="eps",
                        float_value=m.eps,
                    ),
                ),
            ),
            Dependency(
                operation=f"built-in:{_legal_op_type(node.graph._target_to_str(torch.nn.Identity))}",
                argument=(f"::{input_names[0]}",),
                result=(_make_var_name(node.name, suffix="input"),),
            ),
            Dependency(
                operation=f"built-in:{_legal_op_type(node.graph._target_to_str(torch.nn.Identity))}",
                argument=(_make_var_name(node.name, suffix="output"),),
                result=(f"::{output_names[0]}",),
            ),
        ),
        metadata=_nn_module_meta(m),
    )


def dump(
    m: torch.nn.Module,
    *sample_input: torch.Tensor,
    name: str = "",
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
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
                    **_tensor_meta_dict(node.meta["tensor_meta"]),
                )
            )
            if input_names is not None:
                _in = f"::{input_names.pop(0)}"
                dependency.append(
                    Dependency(
                        operation=_legal_op_type(
                            f"built-in:{traced._target_to_str(torch.nn.Identity)}"
                        ),
                        argument=(_in,),
                        result=(_make_var_name(node.name),),
                    )
                )
        elif node.op == "get_attr":  # static inputs
            _p = eval(_torch_qualified_name(f"m.{node.target}"))
            intermediate.append(
                Tensor(
                    name=_make_var_name(node.name),
                    value=[]
                    if omit_value
                    else _p.data.contiguous().view(-1).numpy().tolist(),
                    **_tensor_meta_dict(node.meta["tensor_meta"]),
                )
            )
        elif node.op == "output":  # output
            assert len(node.args) == 1
            output.append(
                Tensor(
                    name=_make_var_name(node.name),
                    **_tensor_meta_dict(node.meta["tensor_meta"]),
                )
            )
            dependency.append(
                Dependency(
                    operation=_legal_op_type(
                        f"built-in:{traced._target_to_str(torch.nn.Identity)}"
                    ),
                    argument=(_make_var_name(node.args[0].name),),
                    result=(
                        _make_var_name(node.name)
                        if output_names is None
                        else f"::{output_names.pop(0)}",
                    ),
                )
            )
        elif node.op in ("call_function", "call_method", "call_module"):  # subgraphs
            intermediate.append(
                Tensor(
                    name=_make_var_name(node.name),
                    **_tensor_meta_dict(node.meta["tensor_meta"]),
                )
            )
            if node.op == "call_module":
                _m = eval(_torch_qualified_name(f"m.{node.target}"))
                _input_names = [_make_var_name(n.__str__()) for n in node.args]
                _output_names = [_make_var_name(node.name)]
                dependency.append(
                    Dependency(
                        operation=node.name,
                        argument=_input_names,
                        result=_output_names,
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
                    if isinstance(_m.sparseness, sparse.Dense):
                        dependency.append(
                            Dependency(
                                operation=_legal_op_type(
                                    f"built-in:{traced._target_to_str(torch.nn.Identity)}"
                                ),
                                argument=(_input_names[0],),
                                result=(_output_names[0],),
                            ),
                        )
                    else:
                        if flat:
                            intermediate.append(
                                Tensor(
                                    name=_make_var_name(node.name, suffix="mask"),
                                    value=[]
                                    if omit_value
                                    else m.mask.data.contiguous()
                                    .view(-1)
                                    .numpy()
                                    .tolist(),
                                    **_tensor_meta_dict(node.meta["tensor_meta"]),
                                ),  # this is a static input
                            )
                            dependency.append(
                                Dependency(
                                    operation=_legal_op_type(
                                        f"built-in:{traced._target_to_str(torch.mul)}"
                                    ),
                                    argument=(
                                        _input_names[0],
                                        _make_var_name(node.name, suffix="mask"),
                                    ),
                                    result=(_output_names[0],),
                                ),
                            )
                        else:
                            subgraph.append(
                                _sparsifier_graph(
                                    _m,
                                    node,
                                    _input_names,
                                    _output_names,
                                    omit_value=omit_value,
                                )
                            )
                elif isinstance(_m, torch.nn.modules.batchnorm._BatchNorm):
                    # manually add a batch_norm op
                    if flat:
                        intermediate.append(
                            Tensor(
                                name=_make_var_name(node.name, suffix="running_mean"),
                                value=[]
                                if omit_value
                                else _m.running_mean.data.contiguous().view(-1).numpy().tolist(),
                                shape=_m.running_mean.shape,
                                format=_legal_format(_m.running_mean.dtype),
                            ),  # this is a static input
                        )
                        intermediate.append(
                            Tensor(
                                name=_make_var_name(node.name, suffix="running_var"),
                                value=[]
                                if omit_value
                                else _m.running_var.data.contiguous().view(-1).numpy().tolist(),
                                shape=_m.running_var.shape,
                                format=_legal_format(_m.running_var.dtype),
                            ),  # this is a static input
                        )
                        intermediate.append(
                            Tensor(
                                name=_make_var_name(node.name, suffix="weight"),
                                value=[]
                                if omit_value
                                else _m.weight.data.contiguous().view(-1).numpy().tolist(),
                                shape=_m.weight.shape,
                                format=_legal_format(_m.weight.dtype),
                            ),  # this is a static input
                        )
                        intermediate.append(
                            Tensor(
                                name=_make_var_name(node.name, suffix="bias"),
                                value=[]
                                if omit_value
                                else _m.bias.data.contiguous().view(-1).numpy().tolist(),
                                shape=_m.bias.shape,
                                format=_legal_format(_m.bias.dtype),
                            ),  # this is a static input
                        )
                        dependency.append(
                            Dependency(
                                operation=f"built-in:{_legal_op_type(node.graph._target_to_str(torch.batch_norm))}",
                                argument=(
                                    _input_names[0],
                                    _make_var_name(node.name, suffix="running_mean"),
                                    _make_var_name(node.name, suffix="running_var"),
                                    _make_var_name(node.name, suffix="weight"),
                                    _make_var_name(node.name, suffix="bias"),
                                ),
                                result=(_output_names[0],),
                                attribute=(
                                    Attribute(
                                        kind=FLOAT,
                                        name="momentum",
                                        float_value=_m.momentum,
                                    ),
                                    Attribute(
                                        kind=FLOAT,
                                        name="eps",
                                        float_value=_m.eps,
                                    ),
                                ),
                            ),
                        )
                    else:
                        subgraph.append(
                            _batch_norm_graph(
                                _m, 
                                node, 
                                _input_names, 
                                _output_names, 
                                omit_value=omit_value,
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
                            input_names=_input_names,
                            output_names=_output_names,
                            flat=flat,
                            omit_value=omit_value,
                            metadata=_nn_module_meta(_m),
                        )
                    )
            else:  # built-in function or tensor method
                dependency.append(
                    Dependency(
                        operation=f"built-in:{traced._target_to_str(node.target)}",
                        argument=itertools.chain(
                            (_make_var_name(n.__str__()) for n in node.args),
                            (
                                _make_var_name(v.__str__())
                                for k, v in node.kwargs.items()
                            ),
                        ),
                        result=(_make_var_name(node.name),),
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


def list_ops(graph: Graph) -> List[str]:
    lot = [
        dep.operation
        for dep in graph.dependency
        if dep.operation.startswith("built-in:")
    ]
    for sg in graph.subgraph:
        lot += list_ops(sg)
    return set(lot)


def save_to_file(model: Graph, filename: str, format="binary") -> None:
    if format == "binary":
        with open(filename, "wb") as f:
            f.write(model.SerializeToString())
    elif format == "json":
        with open(filename, "w") as f:
            f.write(MessageToJson(model))
    else:
        raise ValueError(f"unsupported DMIR file format: {format}")


def load_from_file(filename: str, format="binary") -> Graph:
    if format == "binary":
        graph = Graph()
        with open(filename, "rb") as f:
            graph.ParseFromString(f.read())
    elif format == "json":
        with open(filename, "r") as f:
            graph = Parse(f.read(), Graph())
    else:
        raise ValueError(f"unsupported DMIR file format: {format}")
    return graph


def is_legal(graph, level=0):
    # a checker on whether a DMIR graph meets requirements of a certain level
    # TODO: implement this
    return True


def lower(graph, level=0):
    # a python wrapper of stack transformation of lowering DMIR graph to a level
    # TODO: implement this
    lvl = 0
    while lvl < level and not is_legal(graph, level=lvl):
        graph = lower(graph, level=lvl)
        lvl += 1
    return graph


def executor(graph, level=0):
    # a python wrapper of DMIR C++ executor
    # TODO: implement this
    return None


def cpsim_executor(graph):
    # a python wrapper of CPSIM functional mode applied to DMIR-3 graph
    # TODO: implement this
    return None
