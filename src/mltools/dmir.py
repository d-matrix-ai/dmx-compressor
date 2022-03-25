from mltools.utils.dmir_pb2 import *

# from mltools.utils.graph_to_csv import graph_to_csv
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
import transformers.utils.fx as fx_hf
import transformers
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

        elif isinstance(result, torch.Size):
            n.meta["tensor_meta"] = extract_tensor_metadata(torch.tensor(result))

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
                torch.nn.modules.conv._ConvNd,
                torch.nn.modules.pooling._MaxPoolNd,
                torch.nn.modules.pooling._AdaptiveAvgPoolNd,
                torch.nn.modules.pooling._AvgPoolNd,
                torch.nn.modules.ReLU,
                torch.nn.modules.Linear,
            ),
        )
        if self.flat:
            return is_leaf
        else:
            return (
                is_leaf
                or m.__module__.startswith("torch.nn")
                or m.__module__.startswith("corsair.nn")
            ) and not isinstance(m, torch.nn.Sequential)

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


def _make_var_name(
    name: str, prefix: str = "", suffix: str = "", end: str = "_"
) -> str:
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
        return f"{name}{end}".replace(".", "__")


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
    if isinstance(meta, fx.immutable_collections.immutable_dict):
        meta = meta["start_logits"]

    if isinstance(meta, tuple) and len(meta) == 2:
        meta = meta[0]
        return dict(
            shape=meta.shape,
            format=_legal_format(meta.dtype),
            is_quantized=meta.is_quantized,
            qscheme=str(meta.is_quantized) if meta.is_quantized else "",
            q_scale=meta.q_scale,
            q_zero_point=meta.q_zero_point,
        )

    else:
        return dict(
            shape=meta.shape,
            format=_legal_format(meta.dtype),
            is_quantized=meta.is_quantized,
            qscheme=str(meta.is_quantized) if meta.is_quantized else "",
            q_scale=meta.q_scale,
            q_zero_point=meta.q_zero_point,
        )


def _make_value_for_dumping(x: Optional[Tensor]):
    return x.data.contiguous().view(-1).cpu().numpy().tolist() if x is not None else x


def _sparsifier_graph(m, node, input_names, output_names, omit_value=False):
    return Graph(
        name=node.name,
        input=(
            Tensor(
                name=_make_var_name(node.name, suffix="dense"),
                **_tensor_meta_dict(node.args[0].meta["tensor_meta"]),
            ),  # this is a dynamic input
        ),
        intermediate=(
            Tensor(
                name=_make_var_name(node.name, suffix="mask"),
                value=[] if omit_value else _make_value_for_dumping(m.mask),
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
                operation=f"{_legal_op_type(node.graph._target_to_str(torch.mul))}",
                argument=(
                    _make_var_name(node.name, suffix="dense"),
                    _make_var_name(node.name, suffix="mask"),
                ),
                result=(_make_var_name(node.name, suffix="sparse"),),
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
                value=[] if omit_value else _make_value_for_dumping(m.running_mean),
                shape=m.running_mean.shape,
                format=_legal_format(m.running_mean.dtype),
            ),  # this is a static input
            Tensor(
                name=_make_var_name(node.name, suffix="running_var"),
                value=[] if omit_value else _make_value_for_dumping(m.running_var),
                shape=m.running_var.shape,
                format=_legal_format(m.running_var.dtype),
            ),  # this is a static input
            Tensor(
                name=_make_var_name(node.name, suffix="weight"),
                value=[] if omit_value else _make_value_for_dumping(m.weight),
                shape=m.weight.shape,
                format=_legal_format(m.weight.dtype),
            ),  # this is a static input
            Tensor(
                name=_make_var_name(node.name, suffix="bias"),
                value=[] if omit_value else _make_value_for_dumping(m.bias),
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
                operation=f"{_legal_op_type(node.graph._target_to_str(torch.batch_norm))}",
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
                        kind=Attribute.FLOAT,
                        name="momentum",
                        float_value=m.momentum,
                    ),
                    Attribute(
                        kind=Attribute.FLOAT,
                        name="eps",
                        float_value=m.eps,
                    ),
                ),
            ),
        ),
        metadata=_nn_module_meta(m),
    )


def _linear_graph(m, node, input_names, output_names, omit_value=False):
    _weight = Tensor(
        name=_make_var_name(node.name, suffix="weight"),
        value=[] if omit_value else _make_value_for_dumping(m.weight),
        shape=m.weight.shape,
        format=_legal_format(m.weight.dtype),
    )  # this is a static input
    _bias = (
        Tensor(
            name=_make_var_name(node.name, suffix="bias"),
            value=[] if omit_value else _make_value_for_dumping(m.bias),
            shape=m.bias.shape,
            format=_legal_format(m.bias.dtype),
        )
        if m.bias is not None
        else None
    )  # this is a static input
    return Graph(
        name=node.name,
        input=(
            Tensor(
                name=_make_var_name(node.name, suffix="input"),
                **_tensor_meta_dict(node.args[0].meta["tensor_meta"]),
            ),  # this is a dynamic input
        ),
        intermediate=(_weight,) if _bias is None else (_weight, _bias),
        output=(
            Tensor(
                name=_make_var_name(node.name, suffix="output"),
                **_tensor_meta_dict(node.meta["tensor_meta"]),
            ),
        ),
        dependency=(
            Dependency(
                operation=f"linear",
                argument=(
                    _make_var_name(node.name, suffix="input"),
                    _make_var_name(node.name, suffix="weight"),
                )
                if _bias is None
                else (
                    _make_var_name(node.name, suffix="input"),
                    _make_var_name(node.name, suffix="weight"),
                    _make_var_name(node.name, suffix="bias"),
                ),
                result=(_make_var_name(node.name, suffix="output"),),
                attribute=(),
            ),
        ),
        metadata=_nn_module_meta(m),
    )


def _conv_graph(m, node, input_names, output_names, omit_value=False):
    _weight = Tensor(
        name=_make_var_name(node.name, suffix="weight"),
        value=[] if omit_value else _make_value_for_dumping(m.weight),
        shape=m.weight.shape,
        format=_legal_format(m.weight.dtype),
    )  # this is a static input
    _bias = (
        Tensor(
            name=_make_var_name(node.name, suffix="bias"),
            value=[] if omit_value else _make_value_for_dumping(m.bias),
            shape=m.bias.shape,
            format=_legal_format(m.bias.dtype),
        )
        if m.bias is not None
        else None
    )  # this is a static input
    _input = [
        Tensor(
            name=_make_var_name(node.name, suffix="input"),
            **_tensor_meta_dict(node.args[0].meta["tensor_meta"]),
        ),  # this is a dynamic input
    ]
    _intermediate = [_weight] if _bias is None else [_weight, _bias]
    _output = [
        Tensor(
            name=_make_var_name(node.name, suffix="output"),
            **_tensor_meta_dict(node.meta["tensor_meta"]),
        ),
    ]
    _dependency = [
        Dependency(
            operation=f"conv",
            argument=(
                _make_var_name(node.name, suffix="input"),
                _make_var_name(node.name, suffix="weight"),
            )
            if _bias is None
            else (
                _make_var_name(node.name, suffix="input"),
                _make_var_name(node.name, suffix="weight"),
                _make_var_name(node.name, suffix="bias"),
            ),
            result=(_make_var_name(node.name, suffix="output"),),
            attribute=(
                Attribute(
                    kind=Attribute.INTS,
                    name="stride",
                    integer_values=m.stride,
                )
                if isinstance(m.stride, tuple)
                else Attribute(
                    kind=Attribute.INT,
                    name="stride",
                    integer_value=m.stride,
                ),
                Attribute(
                    kind=Attribute.INTS,
                    name="padding",
                    integer_values=m.padding,
                )
                if isinstance(m.padding, tuple)
                else Attribute(
                    kind=Attribute.INT,
                    name="padding",
                    integer_value=m.padding,
                ),
                Attribute(
                    kind=Attribute.INTS,
                    name="dilation",
                    integer_values=m.dilation,
                )
                if isinstance(m.dilation, tuple)
                else Attribute(
                    kind=Attribute.INT,
                    name="dilation",
                    integer_value=m.dilation,
                ),
                Attribute(
                    kind=Attribute.INT,
                    name="groups",
                    integer_value=m.groups,
                ),
            ),
        ),
    ]
    return Graph(
        name=node.name,
        input=_input,
        intermediate=_intermediate,
        output=_output,
        dependency=_dependency,
        metadata=_nn_module_meta(m),
    )


def _relu_graph(m, node, input_names, output_names, omit_value=False):
    return Graph(
        name=node.name,
        input=(
            Tensor(
                name=_make_var_name(node.name, suffix="input"),
                **_tensor_meta_dict(node.args[0].meta["tensor_meta"]),
            ),  # this is a dynamic input
        ),
        output=(
            Tensor(
                name=_make_var_name(node.name, suffix="output"),
                **_tensor_meta_dict(node.meta["tensor_meta"]),
            ),
        ),
        dependency=(
            Dependency(
                operation=f"relu",
                argument=(_make_var_name(node.name, suffix="input"),),
                result=(_make_var_name(node.name, suffix="output"),),
                attribute=(
                    Attribute(
                        kind=Attribute.INT,
                        name="inplace",
                        integer_value=int(m.inplace),
                    ),
                ),
            ),
        ),
    )


def _max_pool_graph(m, node, input_names, output_names, omit_value=False):
    return Graph(
        name=node.name,
        input=(
            Tensor(
                name=_make_var_name(node.name, suffix="input"),
                **_tensor_meta_dict(node.args[0].meta["tensor_meta"]),
            ),  # this is a dynamic input
        ),
        output=(
            Tensor(
                name=_make_var_name(node.name, suffix="output"),
                **_tensor_meta_dict(node.meta["tensor_meta"]),
            ),
        ),
        dependency=(
            Dependency(
                operation=f"max_pool",
                argument=(_make_var_name(node.name, suffix="input"),),
                result=(_make_var_name(node.name, suffix="output"),),
                attribute=(
                    Attribute(
                        kind=Attribute.INTS,
                        name="kernel_size",
                        integer_values=m.kernel_size,
                    )
                    if isinstance(m.kernel_size, tuple)
                    else Attribute(
                        kind=Attribute.INT,
                        name="kernel_size",
                        integer_value=m.kernel_size,
                    ),
                    Attribute(
                        kind=Attribute.INTS,
                        name="stride",
                        integer_values=m.stride,
                    )
                    if isinstance(m.stride, tuple)
                    else Attribute(
                        kind=Attribute.INT,
                        name="stride",
                        integer_value=m.stride,
                    ),
                    Attribute(
                        kind=Attribute.INTS,
                        name="padding",
                        integer_values=m.padding,
                    )
                    if isinstance(m.padding, tuple)
                    else Attribute(
                        kind=Attribute.INT,
                        name="padding",
                        integer_value=m.padding,
                    ),
                    Attribute(
                        kind=Attribute.INTS,
                        name="dilation",
                        integer_values=m.dilation,
                    )
                    if isinstance(m.dilation, tuple)
                    else Attribute(
                        kind=Attribute.INT,
                        name="dilation",
                        integer_value=m.dilation,
                    ),
                    Attribute(
                        kind=Attribute.INT,
                        name="ceil_mode",
                        integer_value=int(m.ceil_mode),
                    ),
                ),
            ),
        ),
        metadata=_nn_module_meta(m),
    )


def _avg_pool_graph(m, node, input_names, output_names, omit_value=False):
    return Graph(
        name=node.name,
        input=(
            Tensor(
                name=_make_var_name(node.name, suffix="input"),
                **_tensor_meta_dict(node.args[0].meta["tensor_meta"]),
            ),  # this is a dynamic input
        ),
        output=(
            Tensor(
                name=_make_var_name(node.name, suffix="output"),
                **_tensor_meta_dict(node.meta["tensor_meta"]),
            ),
        ),
        dependency=(
            Dependency(
                operation=f"average_pool",
                argument=(_make_var_name(node.name, suffix="input"),),
                result=(_make_var_name(node.name, suffix="output"),),
                attribute=(
                    Attribute(
                        kind=Attribute.INTS,
                        name="kernel_size",
                        integer_values=m.kernel_size,
                    )
                    if isinstance(m.kernel_size, tuple)
                    else Attribute(
                        kind=Attribute.INT,
                        name="kernel_size",
                        integer_value=m.kernel_size,
                    ),
                    Attribute(
                        kind=Attribute.INTS,
                        name="stride",
                        integer_values=m.stride,
                    )
                    if isinstance(m.stride, tuple)
                    else Attribute(
                        kind=Attribute.INT,
                        name="stride",
                        integer_value=m.stride,
                    ),
                    Attribute(
                        kind=Attribute.INTS,
                        name="padding",
                        integer_values=m.padding,
                    )
                    if isinstance(m.padding, tuple)
                    else Attribute(
                        kind=Attribute.INT,
                        name="padding",
                        integer_value=m.padding,
                    ),
                    Attribute(
                        kind=Attribute.INTS,
                        name="dilation",
                        integer_values=m.dilation,
                    )
                    if isinstance(m.dilation, tuple)
                    else Attribute(
                        kind=Attribute.INT,
                        name="dilation",
                        integer_value=m.dilation,
                    ),
                    Attribute(
                        kind=Attribute.INT,
                        name="ceil_mode",
                        integer_value=int(m.ceil_mode),
                    ),
                    Attribute(
                        kind=Attribute.INT,
                        name="count_include_pad",
                        integer_value=int(m.count_include_pad),
                    ),
                ),
            ),
        ),
        metadata=_nn_module_meta(m),
    )


def _adaptive_avg_pool_graph(m, node, input_names, output_names, omit_value=False):
    return Graph(
        name=node.name,
        input=(
            Tensor(
                name=_make_var_name(node.name, suffix="input"),
                **_tensor_meta_dict(node.args[0].meta["tensor_meta"]),
            ),  # this is a dynamic input
        ),
        output=(
            Tensor(
                name=_make_var_name(node.name, suffix="output"),
                **_tensor_meta_dict(node.meta["tensor_meta"]),
            ),
        ),
        dependency=(
            Dependency(
                operation=f"adaptive_average_pool",
                argument=(_make_var_name(node.name, suffix="input"),),
                result=(_make_var_name(node.name, suffix="output"),),
                attribute=(
                    Attribute(
                        kind=Attribute.INTS,
                        name="output_size",
                        integer_values=m.output_size,
                    )
                    if isinstance(m.output_size, tuple)
                    else Attribute(
                        kind=Attribute.INT,
                        name="output_size",
                        integer_value=m.output_size,
                    ),
                ),
            ),
        ),
        metadata=_nn_module_meta(m),
    )


def _list_to_int(x):
    return [x if isinstance(x, int) else x[0]][0]


def parse_fx(
    m: torch.nn.Module,
    *sample_input: torch.Tensor,
    graph_dict: dict = None,
):
    tracer = DMIRTracer()
    gm = fx.GraphModule(root=m, graph=tracer.trace(m))
    ShapeProp(gm).propagate(*sample_input)
    traced = gm.graph

    for node in traced.nodes:
        op_name = node.name
        input_format, output_format, input_shape, output_shape = None, None, None, None
        weight_format, weight_shape, weight_sparsity = None, None, None
        padding, stride, kernel_size = None, None, None

        _input_names = [_make_var_name(n.__str__(), end="") for n in node.args]
        _output_names = [_make_var_name(node.name, end="")]

        if node.op == "placeholder":  # dynamic inputs
            op_name = "input"

        elif node.op == "output":
            op_name = "output"

        elif node.op == "get_attr":  # static inputs
            pass  # .T (transpose) op applied in the forward call can be processed here

        elif node.op in ("call_function", "call_method", "call_module"):  # subgraphs

            if node.op == "call_module":
                _m = eval(_torch_qualified_name(f"m.{node.target}"))
                if isinstance(_m, torch.nn.modules.conv._ConvNd) or isinstance(
                    _m, torch.nn.modules.linear.Linear
                ):
                    if isinstance(_m, torch.nn.modules.conv._ConvNd):
                        op_name = "conv"
                        padding = _list_to_int(_m.padding)
                        stride = _list_to_int(_m.stride)
                        kernel_size = _list_to_int(_m.kernel_size)
                    elif isinstance(_m, torch.nn.modules.linear.Linear):
                        op_name = "linear"

                    weight_shape = list(_m.weight.shape)

                    if isinstance(_m.weight_cast.format, numerical.BlockFloatingPoint):
                        if _m.weight_cast.format.precision == 4:
                            weight_format = "BFP12"
                        elif _m.weight_cast.format.precision == 8:
                            weight_format = "BFP16"
                    else:
                        weight_format = "FP16"

                    if isinstance(_m.input_cast.format, numerical.BlockFloatingPoint):
                        if _m.input_cast.format.precision == 4:
                            input_format = "BFP12"
                        elif _m.input_cast.format.precision == 8:
                            input_format = "BFP16"
                    else:
                        input_format = "FP16"

                    if isinstance(_m.weight_sparsifier.sparseness, sparse.Dense):
                        weight_sparsity = "dense"
                    else:
                        weight_sparsity = "sparse"

                elif isinstance(_m, torch.nn.modules.pooling._MaxPoolNd):
                    op_name = "max_pool"
                    padding = _list_to_int(_m.padding)
                    stride = _list_to_int(_m.stride)
                    kernel_size = _list_to_int(_m.kernel_size)
                elif isinstance(_m, torch.nn.modules.pooling._AvgPoolNd):
                    op_name = "avg_pool"
                    padding = _list_to_int(_m.padding)
                    stride = _list_to_int(_m.stride)
                    kernel_size = _list_to_int(_m.kernel_size)
                elif isinstance(_m, torch.nn.modules.pooling._AdaptiveAvgPoolNd):
                    op_name = "global_avg_pool"
                elif isinstance(_m, torch.nn.modules.ReLU):
                    op_name = "relu"
                elif isinstance(_m, torch.nn.modules.batchnorm._BatchNorm):
                    op_name = "batchnorm"

                else:  # custom modules
                    subgraph_sample_input = [
                        torch.randn(
                            arg.meta["tensor_meta"].shape,
                            dtype=arg.meta["tensor_meta"].dtype,
                            device=device,
                        )
                        for arg in node.args
                    ]
                    parse_fx(
                        _m,
                        *subgraph_sample_input,
                        graph_dict=graph_dict,
                    )
            else:  # built-in function or tensor method
                op_name = f"{traced._target_to_str(node.target)}"
                if op_name == "matmul":
                    weight_shape = list(node.args[1].meta["tensor_meta"].shape)
                    weight_format = "FP16"

        else:
            raise RuntimeError(f"illegal FXIR node opcode {node.op}")

        if node.args != ():
            input_shape = list(node.args[0].meta["tensor_meta"].shape)
            output_shape = list(node.meta["tensor_meta"].shape)
            if input_format is None:
                input_format = "FP16"
            if output_format is None:
                output_format = "FP16"

        graph_dict[node.name] = {
            "op": op_name,
            "input_name": _input_names,
            "input_shape": input_shape,
            "input_format": input_format,
            "param_shape": weight_shape,
            "param_format": weight_format,
            "param_sparsity": weight_sparsity,
            "output_name": _output_names,
            "output_shape": output_shape,
            "output_format": output_format,
            "padding": padding,
            "stride": stride,
            "kernel_size": kernel_size,
        }
    return graph_dict


def dump(
    m: torch.nn.Module,
    *sample_input: torch.Tensor,  # TODO type check for HF input
    name: str = "",
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    flat: bool = False,
    omit_value: bool = False,
    metadata: str = "",
) -> Graph:

    tracer = DMIRTracer(flat=flat)

    if isinstance(m, transformers.models.bert.modeling_bert.BertForQuestionAnswering):
        gm = fx_hf.symbolic_trace(
            m,
            batch_size=1,
            sequence_length=384,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
        )

        sample_input = sample_input[0]
        ShapeProp(gm).propagate(
            sample_input["input_ids"],
            sample_input["attention_mask"],
            sample_input["token_type_ids"],
        )
        device = sample_input["input_ids"].device

    else:
        graph = tracer.trace(m)
        gm = fx.GraphModule(root=m, graph=graph)
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
        elif node.op == "get_attr":  # static inputs
            _p = eval(_torch_qualified_name(f"m.{node.target}"))
            intermediate.append(
                Tensor(
                    name=_make_var_name(node.name),
                    value=[] if omit_value else _make_value_for_dumping(_p),
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
                        f"{traced._target_to_str(torch.nn.Identity)}"
                    ),
                    argument=(
                        _make_var_name(
                            node.args[0]["start_logits"].name
                            if isinstance(
                                node.args[0], fx.immutable_collections.immutable_dict
                            )
                            else node.args[0].name
                        ),
                    ),
                    result=(_make_var_name(node.name),),
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
                if isinstance(_m, numerical.CastTo):
                    dependency.append(
                        Dependency(
                            operation=node.name,
                            argument=_input_names,
                            result=_output_names,
                            attribute=_corsair_specific_attributes(_m),
                        )
                    )
                    subgraph.append(
                        Graph(
                            name=node.name,
                            op_type=_legal_op_type("cast_to"),
                            metadata=_nn_module_meta(_m),
                        )
                    )
                elif isinstance(_m, sparse.Sparsify):
                    dependency.append(
                        Dependency(
                            operation=node.name,
                            argument=_input_names,
                            result=_output_names,
                            attribute=_corsair_specific_attributes(_m),
                        )
                    )
                    if isinstance(_m.sparseness, sparse.Dense):
                        dependency.append(
                            Dependency(
                                operation=_legal_op_type(
                                    f"{traced._target_to_str(torch.nn.Identity)}"
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
                                    else _make_value_for_dumping(m.mask),
                                    **_tensor_meta_dict(node.meta["tensor_meta"]),
                                ),  # this is a static input
                            )
                            dependency.append(
                                Dependency(
                                    operation=_legal_op_type(
                                        f"{traced._target_to_str(torch.mul)}"
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
                                else _make_value_for_dumping(_m.running_mean),
                                shape=_m.running_mean.shape,
                                format=_legal_format(_m.running_mean.dtype),
                            ),  # this is a static input
                        )
                        intermediate.append(
                            Tensor(
                                name=_make_var_name(node.name, suffix="running_var"),
                                value=[]
                                if omit_value
                                else _make_value_for_dumping(_m.running_var),
                                shape=_m.running_var.shape,
                                format=_legal_format(_m.running_var.dtype),
                            ),  # this is a static input
                        )
                        intermediate.append(
                            Tensor(
                                name=_make_var_name(node.name, suffix="weight"),
                                value=[]
                                if omit_value
                                else _make_value_for_dumping(_m.weight),
                                shape=_m.weight.shape,
                                format=_legal_format(_m.weight.dtype),
                            ),  # this is a static input
                        )
                        intermediate.append(
                            Tensor(
                                name=_make_var_name(node.name, suffix="bias"),
                                value=[]
                                if omit_value
                                else _make_value_for_dumping(_m.bias),
                                shape=_m.bias.shape,
                                format=_legal_format(_m.bias.dtype),
                            ),  # this is a static input
                        )
                        dependency.append(
                            Dependency(
                                operation=f"{_legal_op_type(node.graph._target_to_str(torch.batch_norm))}",
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
                                        kind=Attribute.FLOAT,
                                        name="momentum",
                                        float_value=_m.momentum,
                                    ),
                                    Attribute(
                                        kind=Attribute.FLOAT,
                                        name="eps",
                                        float_value=_m.eps,
                                    ),
                                ),
                            ),
                        )
                    else:
                        dependency.append(
                            Dependency(
                                operation=node.name,
                                argument=_input_names,
                                result=_output_names,
                                attribute=_corsair_specific_attributes(_m),
                            )
                        )
                        subgraph.append(
                            _batch_norm_graph(
                                _m,
                                node,
                                _input_names,
                                _output_names,
                                omit_value=omit_value,
                            )
                        )
                elif isinstance(_m, torch.nn.modules.Linear):
                    _weight = Tensor(
                        name=_make_var_name(node.name, suffix="weight"),
                        value=[] if omit_value else _make_value_for_dumping(_m.weight),
                        shape=_m.weight.shape,
                        format=_legal_format(_m.weight.dtype),
                    )  # this is a static input
                    _bias = (
                        Tensor(
                            name=_make_var_name(node.name, suffix="bias"),
                            value=[]
                            if omit_value
                            else _make_value_for_dumping(_m.bias),
                            shape=_m.bias.shape,
                            format=_legal_format(_m.bias.dtype),
                        )
                        if _m.bias is not None
                        else None
                    )  # this is a static input
                    if flat:
                        intermediate.append(_weight)
                        if _bias is not None:
                            intermediate.append(_bias)
                        dependency.append(
                            Dependency(
                                operation=f"linear",
                                argument=(
                                    _input_names[0],
                                    _make_var_name(node.name, suffix="weight"),
                                )
                                if _bias is None
                                else (
                                    _input_names[0],
                                    _make_var_name(node.name, suffix="weight"),
                                    _make_var_name(node.name, suffix="bias"),
                                ),
                                result=(_output_names[0],),
                                attribute=(),
                            ),
                        )
                    else:
                        dependency.append(
                            Dependency(
                                operation=node.name,
                                argument=_input_names,
                                result=_output_names,
                                attribute=_corsair_specific_attributes(_m),
                            )
                        )
                        if isinstance(_m, corsair.nn.Linear):
                            subgraph.append(
                                dump(
                                    _m,
                                    *[
                                        torch.randn(
                                            arg.meta["tensor_meta"].shape,
                                            dtype=arg.meta["tensor_meta"].dtype,
                                            device=_m.weight.device,
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
                        else:
                            subgraph.append(
                                _linear_graph(
                                    _m,
                                    node,
                                    _input_names,
                                    _output_names,
                                    omit_value=omit_value,
                                )
                            )
                elif isinstance(_m, torch.nn.modules.conv._ConvNd):
                    _weight = Tensor(
                        name=_make_var_name(node.name, suffix="weight"),
                        value=[] if omit_value else _make_value_for_dumping(_m.weight),
                        shape=_m.weight.shape,
                        format=_legal_format(_m.weight.dtype),
                    )  # this is a static input
                    _bias = (
                        Tensor(
                            name=_make_var_name(node.name, suffix="bias"),
                            value=[]
                            if omit_value
                            else _make_value_for_dumping(_m.bias),
                            shape=_m.bias.shape,
                            format=_legal_format(_m.bias.dtype),
                        )
                        if _m.bias is not None
                        else None
                    )  # this is a static input
                    if flat:
                        intermediate.append(_weight)
                        if _bias is not None:
                            intermediate.append(_bias)
                        dependency.append(
                            Dependency(
                                operation=f"conv",
                                argument=(
                                    _input_names[0],
                                    _make_var_name(node.name, suffix="weight"),
                                )
                                if _bias is None
                                else (
                                    _input_names[0],
                                    _make_var_name(node.name, suffix="weight"),
                                    _make_var_name(node.name, suffix="bias"),
                                ),
                                result=(_output_names[0],),
                                attribute=(
                                    Attribute(
                                        kind=Attribute.INTS,
                                        name="stride",
                                        integer_values=_m.stride,
                                    )
                                    if isinstance(_m.stride, tuple)
                                    else Attribute(
                                        kind=Attribute.INT,
                                        name="stride",
                                        integer_value=_m.stride,
                                    ),
                                    Attribute(
                                        kind=Attribute.INTS,
                                        name="padding",
                                        integer_values=_m.padding,
                                    )
                                    if isinstance(_m.padding, tuple)
                                    else Attribute(
                                        kind=Attribute.INT,
                                        name="padding",
                                        integer_value=_m.padding,
                                    ),
                                    Attribute(
                                        kind=Attribute.INTS,
                                        name="dilation",
                                        integer_values=_m.dilation,
                                    )
                                    if isinstance(_m.dilation, tuple)
                                    else Attribute(
                                        kind=Attribute.INT,
                                        name="dilation",
                                        integer_value=_m.dilation,
                                    ),
                                    Attribute(
                                        kind=Attribute.INT,
                                        name="groups",
                                        integer_value=_m.groups,
                                    ),
                                ),
                            ),
                        )
                    else:
                        dependency.append(
                            Dependency(
                                operation=node.name,
                                argument=_input_names,
                                result=_output_names,
                                attribute=_corsair_specific_attributes(_m),
                            )
                        )
                        if isinstance(_m, corsair.nn.Conv2d):
                            subgraph.append(
                                dump(
                                    _m,
                                    *[
                                        torch.randn(
                                            arg.meta["tensor_meta"].shape,
                                            dtype=arg.meta["tensor_meta"].dtype,
                                            device=_m.weight.device,
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
                        else:
                            subgraph.append(
                                _conv_graph(
                                    _m,
                                    node,
                                    _input_names,
                                    _output_names,
                                    omit_value=omit_value,
                                )
                            )
                elif isinstance(_m, torch.nn.modules.pooling._MaxPoolNd):
                    if flat:
                        dependency.append(
                            Dependency(
                                operation=f"max_pool",
                                argument=(_input_names[0],),
                                result=(_output_names[0],),
                                attribute=(
                                    Attribute(
                                        kind=Attribute.INTS,
                                        name="kernel_size",
                                        integer_values=_m.kernel_size,
                                    )
                                    if isinstance(_m.kernel_size, tuple)
                                    else Attribute(
                                        kind=Attribute.INT,
                                        name="kernel_size",
                                        integer_value=_m.kernel_size,
                                    ),
                                    Attribute(
                                        kind=Attribute.INTS,
                                        name="stride",
                                        integer_values=_m.stride,
                                    )
                                    if isinstance(_m.stride, tuple)
                                    else Attribute(
                                        kind=Attribute.INT,
                                        name="stride",
                                        integer_value=_m.stride,
                                    ),
                                    Attribute(
                                        kind=Attribute.INTS,
                                        name="padding",
                                        integer_values=_m.padding,
                                    )
                                    if isinstance(_m.padding, tuple)
                                    else Attribute(
                                        kind=Attribute.INT,
                                        name="padding",
                                        integer_value=_m.padding,
                                    ),
                                    Attribute(
                                        kind=Attribute.INTS,
                                        name="dilation",
                                        integer_values=_m.dilation,
                                    )
                                    if isinstance(_m.dilation, tuple)
                                    else Attribute(
                                        kind=Attribute.INT,
                                        name="dilation",
                                        integer_value=_m.dilation,
                                    ),
                                    Attribute(
                                        kind=Attribute.INT,
                                        name="ceil_mode",
                                        integer_value=int(_m.ceil_mode),
                                    ),
                                ),
                            )
                        )
                    else:
                        dependency.append(
                            Dependency(
                                operation=node.name,
                                argument=_input_names,
                                result=_output_names,
                                attribute=_corsair_specific_attributes(_m),
                            )
                        )
                        subgraph.append(
                            _max_pool_graph(
                                _m,
                                node,
                                _input_names,
                                _output_names,
                                omit_value=omit_value,
                            )
                        )
                elif isinstance(_m, torch.nn.modules.pooling._AvgPoolNd):
                    if flat:
                        dependency.append(
                            Dependency(
                                operation=f"average_pool",
                                argument=(_input_names[0],),
                                result=(_output_names[0],),
                                attribute=(
                                    Attribute(
                                        kind=Attribute.INTS,
                                        name="kernel_size",
                                        integer_values=_m.kernel_size,
                                    )
                                    if isinstance(_m.kernel_size, tuple)
                                    else Attribute(
                                        kind=Attribute.INT,
                                        name="kernel_size",
                                        integer_value=_m.kernel_size,
                                    ),
                                    Attribute(
                                        kind=Attribute.INTS,
                                        name="stride",
                                        integer_values=_m.stride,
                                    )
                                    if isinstance(_m.stride, tuple)
                                    else Attribute(
                                        kind=Attribute.INT,
                                        name="stride",
                                        integer_value=_m.stride,
                                    ),
                                    Attribute(
                                        kind=Attribute.INTS,
                                        name="padding",
                                        integer_values=_m.padding,
                                    )
                                    if isinstance(_m.padding, tuple)
                                    else Attribute(
                                        kind=Attribute.INT,
                                        name="padding",
                                        integer_value=_m.padding,
                                    ),
                                    Attribute(
                                        kind=Attribute.INTS,
                                        name="dilation",
                                        integer_values=_m.dilation,
                                    )
                                    if isinstance(_m.dilation, tuple)
                                    else Attribute(
                                        kind=Attribute.INT,
                                        name="dilation",
                                        integer_value=_m.dilation,
                                    ),
                                    Attribute(
                                        kind=Attribute.INT,
                                        name="ceil_mode",
                                        integer_value=int(_m.ceil_mode),
                                    ),
                                    Attribute(
                                        kind=Attribute.INT,
                                        name="count_include_pad",
                                        integer_value=int(_m.count_include_pad),
                                    ),
                                ),
                            )
                        )
                    else:
                        dependency.append(
                            Dependency(
                                operation=node.name,
                                argument=_input_names,
                                result=_output_names,
                                attribute=_corsair_specific_attributes(_m),
                            )
                        )
                        subgraph.append(
                            _avg_pool_graph(
                                _m,
                                node,
                                _input_names,
                                _output_names,
                                omit_value=omit_value,
                            )
                        )
                elif isinstance(_m, torch.nn.modules.pooling._AdaptiveAvgPoolNd):
                    if flat:
                        dependency.append(
                            Dependency(
                                operation=f"adaptive_average_pool",
                                argument=(_input_names[0],),
                                result=(_output_names[0],),
                                attribute=(
                                    Attribute(
                                        kind=Attribute.INTS,
                                        name="output_size",
                                        integer_values=_m.output_size,
                                    )
                                    if isinstance(_m.output_size, tuple)
                                    else Attribute(
                                        kind=Attribute.INT,
                                        name="output_size",
                                        integer_value=_m.output_size,
                                    ),
                                ),
                            )
                        )
                    else:
                        dependency.append(
                            Dependency(
                                operation=node.name,
                                argument=_input_names,
                                result=_output_names,
                                attribute=_corsair_specific_attributes(_m),
                            )
                        )
                        subgraph.append(
                            _adaptive_avg_pool_graph(
                                _m,
                                node,
                                _input_names,
                                _output_names,
                                omit_value=omit_value,
                            )
                        )
                elif isinstance(_m, torch.nn.modules.ReLU):
                    if flat:
                        dependency.append(
                            Dependency(
                                operation=f"{_legal_op_type(node.graph._target_to_str(torch.relu))}",
                                argument=(_input_names[0],),
                                result=(_output_names[0],),
                                attribute=(
                                    Attribute(
                                        kind=Attribute.INT,
                                        name="inplace",
                                        integer_value=int(_m.inplace),
                                    ),
                                ),
                            ),
                        )
                    else:
                        dependency.append(
                            Dependency(
                                operation=node.name,
                                argument=_input_names,
                                result=_output_names,
                                attribute=_corsair_specific_attributes(_m),
                            )
                        )
                        subgraph.append(
                            _relu_graph(
                                _m,
                                node,
                                _input_names,
                                _output_names,
                                omit_value=omit_value,
                            )
                        )
                else:  # custom modules
                    dependency.append(
                        Dependency(
                            operation=node.name,
                            argument=_input_names,
                            result=_output_names,
                            attribute=_corsair_specific_attributes(_m),
                        )
                    )
                    subgraph.append(
                        dump(
                            _m,
                            *[
                                torch.zeros(
                                    arg.meta["tensor_meta"].shape,
                                    dtype=arg.meta["tensor_meta"].dtype,
                                    device=device,
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
                if node.target == torch.flatten:
                    start_dim = node.args[1] if len(node.args) > 1 else 0
                    end_dim = node.args[2] if len(node.args) > 2 else -1
                    node.args = (node.args[0],)
                    dependency.append(
                        Dependency(
                            operation=f"{traced._target_to_str(node.target)}",
                            argument=itertools.chain(
                                (_make_var_name(n.__str__()) for n in node.args),
                                (
                                    _make_var_name(v.__str__())
                                    for k, v in node.kwargs.items()
                                ),
                            ),
                            result=(_make_var_name(node.name),),
                            attribute=(
                                Attribute(
                                    kind=Attribute.INT,
                                    name="start_dim",
                                    integer_value=start_dim,
                                ),
                                Attribute(
                                    kind=Attribute.INT,
                                    name="end_dim",
                                    integer_value=end_dim,
                                ),
                            ),
                        )
                    )
                elif node.target == torch.nn.functional.conv2d:
                    (
                        _input,
                        _weight,
                        _bias,
                        _stride,
                        _padding,
                        _dilation,
                        _groups,
                    ) = node.args
                    dependency.append(
                        Dependency(
                            operation=f"conv",
                            argument=(
                                _make_var_name(_input.name),
                                _make_var_name(_weight.name),
                            )
                            if _bias is None
                            else (
                                _make_var_name(_input.name),
                                _make_var_name(_weight.name),
                                _make_var_name(_bias.name),
                            ),
                            result=(_make_var_name(node.name),),
                            attribute=(
                                Attribute(
                                    kind=Attribute.INTS,
                                    name="stride",
                                    integer_values=_stride,
                                )
                                if isinstance(_stride, tuple)
                                else Attribute(
                                    kind=Attribute.INT,
                                    name="stride",
                                    integer_value=_stride,
                                ),
                                Attribute(
                                    kind=Attribute.INTS,
                                    name="padding",
                                    integer_values=_padding,
                                )
                                if isinstance(_padding, tuple)
                                else Attribute(
                                    kind=Attribute.INT,
                                    name="padding",
                                    integer_value=_padding,
                                ),
                                Attribute(
                                    kind=Attribute.INTS,
                                    name="dilation",
                                    integer_values=_dilation,
                                )
                                if isinstance(_dilation, tuple)
                                else Attribute(
                                    kind=Attribute.INT,
                                    name="dilation",
                                    integer_value=_dilation,
                                ),
                                Attribute(
                                    kind=Attribute.INT,
                                    name="groups",
                                    integer_value=_groups,
                                ),
                            ),
                        ),
                    )
                elif (
                    node.target == torch.unsqueeze
                    or node.target == "unsqueeze"
                    or node.target == torch.squeeze
                    or node.target == "squeeze"
                ):
                    dim = node.args[1] if len(node.args) > 1 else None
                    dependency.append(
                        Dependency(
                            operation=f"{traced._target_to_str(node.target)}",
                            argument=(_make_var_name(node.args[0].name),),
                            result=(_make_var_name(node.name),),
                            attribute=(
                                Attribute(
                                    kind=Attribute.INT,
                                    name="dim",
                                    integer_value=dim,
                                ),
                            )
                            if dim is not None
                            else (),
                        )
                    )
                elif (
                    node.target == "view"
                    or node.target == torch.reshape
                    or node.target == "reshape"
                ):
                    shape = node.args[1:]
                    dependency.append(
                        Dependency(
                            operation=f"{traced._target_to_str(node.target)}",
                            argument=(_make_var_name(node.args[0].name),),
                            result=(_make_var_name(node.name),),
                            attribute=(
                                Attribute(
                                    kind=Attribute.INTS,
                                    name="shape",
                                    integer_values=shape,
                                ),
                            ),
                        )
                    )
                elif node.target == "transpose" or node.target == torch.transpose:
                    dim0, dim1 = node.args[1], node.args[2]
                    dependency.append(
                        Dependency(
                            operation=f"{traced._target_to_str(node.target)}",
                            argument=(_make_var_name(node.args[0].name),),
                            result=(_make_var_name(node.name),),
                            attribute=(
                                Attribute(
                                    kind=Attribute.INT,
                                    name="dim0",
                                    integer_value=dim0,
                                ),
                                Attribute(
                                    kind=Attribute.INT,
                                    name="dim1",
                                    integer_value=dim1,
                                ),
                            ),
                        )
                    )
                elif node.target == "permute" or node.target == torch.permute:
                    dims = node.args[1:]
                    dependency.append(
                        Dependency(
                            operation=f"{traced._target_to_str(node.target)}",
                            argument=(_make_var_name(node.args[0].name),),
                            result=(_make_var_name(node.name),),
                            attribute=(
                                Attribute(
                                    kind=Attribute.INTS,
                                    name="dim",
                                    integer_values=dims,
                                ),
                            ),
                        )
                    )
                elif (
                    node.target == "softmax"
                    or node.target == torch.nn.functional.softmax
                ):
                    dim = node.args[1] if len(node.args) > 1 else -1
                    dependency.append(
                        Dependency(
                            operation=f"{traced._target_to_str(node.target)}",
                            argument=(_make_var_name(node.args[0].name),),
                            result=(_make_var_name(node.name),),
                            attribute=(
                                Attribute(
                                    kind=Attribute.INT,
                                    name="dim",
                                    integer_value=-1 if dim is None else dim,
                                ),
                            ),
                        )
                    )
                elif (
                    node.target == "layer_norm"
                    or node.target == torch.nn.functional.layer_norm
                ):
                    dependency.append(
                        Dependency(
                            operation=f"{traced._target_to_str(node.target)}",
                            argument=(
                                _make_var_name(node.args[0].name),
                                _make_var_name(node.kwargs["weight"].name),
                                _make_var_name(node.kwargs["bias"].name),
                            ),
                            result=(_make_var_name(node.name),),
                            attribute=(
                                Attribute(
                                    kind=Attribute.INTS,
                                    name="normalized_shape",
                                    integer_values=node.args[1],
                                ),
                                Attribute(
                                    kind=Attribute.FLOAT,
                                    name="eps",
                                    float_value=node.kwargs["eps"],
                                ),
                            ),
                        )
                    )
                elif (
                    node.target == "embedding"
                    or node.target == torch.nn.functional.embedding
                ):
                    attrs = []
                    if node.args[2] is not None:
                        attrs.append(
                            Attribute(
                                kind=Attribute.INT,
                                name="padding_idx",
                                integer_value=node.args[2],
                            )
                        )
                    if node.args[3] is not None:
                        attrs.append(
                            Attribute(
                                kind=Attribute.FLOAT,
                                name="max_norm",
                                float_value=node.args[3],
                            )
                        )
                    attrs.append(
                        Attribute(
                            kind=Attribute.FLOAT,
                            name="norm_type",
                            float_value=node.args[4],
                        )
                    )
                    attrs.append(
                        Attribute(
                            kind=Attribute.INT,
                            name="scale_grad_by_freq",
                            integer_value=int(node.args[5]),
                        )
                    )
                    attrs.append(
                        Attribute(
                            kind=Attribute.INT,
                            name="sparse",
                            integer_value=int(node.args[6]),
                        )
                    )
                    dependency.append(
                        Dependency(
                            operation=f"{traced._target_to_str(node.target)}",
                            argument=(
                                _make_var_name(node.args[0].name),
                                _make_var_name(node.args[1].name),
                            ),
                            result=(_make_var_name(node.name),),
                            attribute=attrs,
                        )
                    )
                elif (
                    node.target == "dropout"
                    or node.target == torch.nn.functional.dropout
                ):
                    dependency.append(
                        Dependency(
                            operation=f"{traced._target_to_str(node.target)}",
                            argument=(_make_var_name(node.args[0].name),),
                            result=(_make_var_name(node.name),),
                            attribute=(
                                Attribute(
                                    kind=Attribute.FLOAT,
                                    name="p",
                                    float_value=node.kwargs["p"],
                                ),
                                Attribute(
                                    kind=Attribute.INT,
                                    name="training",
                                    integer_value=int(node.kwargs["training"]),
                                ),
                                Attribute(
                                    kind=Attribute.INT,
                                    name="inplace",
                                    integer_value=int(node.kwargs["inplace"]),
                                ),
                            ),
                        )
                    )
                else:
                    dependency.append(
                        Dependency(
                            operation=f"{traced._target_to_str(node.target)}",
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
    lot = [dep.operation for dep in graph.dependency]
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
