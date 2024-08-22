import copy
import functools
import itertools
import operator
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
)

from mltools.numerical.format import Format
default_format = Format.from_shorthand("BFP[8|8]{64,-1}(SN)")

import torch
import torch.nn.functional as F
from pt2bfp.fake_quantize import (
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
)
from pt2bfp.observer import (
    HistogramObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)
from pt2bfp.pt2e.graph_utils import find_sequential_partitions
from pt2bfp.quantizer import (
    BFPQuantizationAnnotation,
    BFPQuantizationSpec,
    BFPQuantizer,
    SharedBFPQuantizationSpec,
)

from pt2bfp.utils import (
    _is_annotated,
    BFPOperatorConfig,
    BFPQuantizationConfig,
    get_bias_qspec,
    get_input_act_qspec,
    get_output_act_qspec,
    get_weight_qspec,
    OperatorPatternType,
)

from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import (
    get_source_partitions,
    SourcePartition,
)

if TYPE_CHECKING:
    from pt2bfp.qconfig import _ObserverConstructor

__all__ = [
    "X86InductorQuantizer",
    "get_default_x86_inductor_quantization_config",
]


@dataclass
class _X86InductorQuantizationAnnotation(BFPQuantizationAnnotation):
    # _is_output_of_quantized_pattern:
    #  * Node as output node of a fusion pattern.
    #  * The fusion pattern supports int8 data type.
    #  * The fusion pattern has inputs annotated to insert observer.
    _is_output_of_quantized_pattern: bool = False


# Operators that:
# 1. Operators are optimized to run with int8 when int8 input provided.
# 2. Operators do not support int8 input and produce fp32 output.
int8_in_int8_out_ops: Set = {
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.cat.default,
    torch.ops.aten.avg_pool2d.default,
    torch.ops.aten.adaptive_avg_pool2d.default,
    torch.ops.aten.flatten.using_ints,
}

# Operators that support the int8 data type for quantization config propagation.
# A superset of int8_in_int8_out_ops incorporating additional operators.
propagation_quantizable_ops = int8_in_int8_out_ops

# Operators support the int8 data type
# and recipe is configured by default in X86InductorQuantizer.
default_quantizable_ops = propagation_quantizable_ops | {
    torch.ops.aten.conv2d.default,
    torch.ops.aten.linear.default,
}

# A superset of default_quantizable_ops includes operators support the int8 data type
# but not enabled by default recipe of X86InductorQuantizer.
quantizable_ops = default_quantizable_ops | {
    torch.ops.aten.matmul.default,
}

QUANT_ANNOTATION_KEY = "quantization_annotation"


def _map_module_function_to_aten_operator_type():
    module_function_to_aten_operator: Dict[Callable, torch._ops.OpOverloadPacket] = {}
    map_list = (
        ([torch.nn.Conv2d, F.conv2d], torch.ops.aten.conv2d.default),
        ([torch.nn.Linear, F.linear], torch.ops.aten.linear.default),
        ([torch.nn.MaxPool2d, F.max_pool2d], torch.ops.aten.max_pool2d.default),
        (
            [
                torch.cat,
            ],
            torch.ops.aten.cat.default,
        ),
        ([torch.nn.AvgPool2d, F.avg_pool2d], torch.ops.aten.avg_pool2d.default),
        (
            [torch.nn.AdaptiveAvgPool2d, F.adaptive_avg_pool2d],
            torch.ops.aten.adaptive_avg_pool2d.default,
        ),
        (
            [
                torch.flatten,
            ],
            torch.ops.aten.flatten.using_ints,
        ),
        (
            [
                torch.matmul,
            ],
            torch.ops.aten.matmul.default,
        ),
    )
    for map_item in map_list:
        module_function_to_aten_operator.update(dict.fromkeys(map_item[0], map_item[1]))  # type: ignore[call-overload]
    return module_function_to_aten_operator


def _mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if node is not None:
            if QUANT_ANNOTATION_KEY not in node.meta:
                node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation()
            node.meta[QUANT_ANNOTATION_KEY]._annotated = True


def _is_node_annotated(_node):
    """
    return True if the node is annotated, otherwise return False
    """
    return (
        QUANT_ANNOTATION_KEY in _node.meta
        and _node.meta[QUANT_ANNOTATION_KEY]._annotated
    )


def _is_any_annotated(nodes: List[Node]):
    """
    Given a list of nodes (that represents an operator pattern),
    check if any of the node is annotated, return True if any of the node
    is annotated, otherwise return False.
    """
    return any(_is_node_annotated(node) for node in nodes)


def _is_all_annotated(nodes: List[Node]):
    """
    Given a list of nodes (that represents an operator pattern),
    return True if all of the node is annotated, otherwise return False.
    """
    return all(_is_node_annotated(node) for node in nodes)


def _is_quantized_op_pt2e(node: torch.fx.Node):
    """
    Used for pt2e flow to check if the node is a quantized node:
    Case1: the node has been annotated as output node of a fusion pattern.
    Case2: the node has been annotated as single quantized node.
    """
    if not _is_any_annotated([node]):
        # The node has not been annotated, directly return False
        return False
    quantization_annotation = node.meta.get(QUANT_ANNOTATION_KEY, None)
    assert isinstance(quantization_annotation, _X86InductorQuantizationAnnotation)
    return quantization_annotation._is_output_of_quantized_pattern


def _supported_quantized_operators() -> Dict[str, List[OperatorPatternType]]:
    # TODO: Add more supported operators here.
    supported_operators: Dict[str, List[OperatorPatternType]] = {
        "conv2d": [
            [torch.nn.Conv2d],
            [F.conv2d],
        ],
    }

    # Append Conv Optional(Add) Optioinal(ReLU)
    conv_add_relu_options = itertools.product(
        [torch.nn.Conv2d, F.conv2d],
        [torch.add, operator.add, None],  # add
        [torch.nn.ReLU, F.relu, None],  # relu
    )
    for conv_op, add_op, relu_op in conv_add_relu_options:
        if add_op is None:
            # Append Conv ReLU
            supported_operators["conv2d"].append([conv_op, relu_op])  # type: ignore[list-item]
        elif relu_op is None:
            # Append Conv Add
            supported_operators["conv2d"].append([conv_op, add_op])  # type: ignore[list-item]
        else:
            # Append Conv Add ReLU
            supported_operators["conv2d"].append([conv_op, add_op, relu_op])  # type: ignore[list-item]

    return copy.deepcopy(supported_operators)


def _get_supported_x86_inductor_config_and_operators() -> List[BFPOperatorConfig]:
    supported_config_and_operators: List[BFPOperatorConfig] = []
    for quantization_config in [
        get_default_x86_inductor_quantization_config(),
    ]:
        ops = _supported_quantized_operators()
        for pattern_list in ops.values():
            supported_config_and_operators.append(
                BFPOperatorConfig(quantization_config, pattern_list)
            )
    return copy.deepcopy(supported_config_and_operators)


@functools.lru_cache
def get_default_x86_inductor_quantization_config(
    is_qat: bool = False,
    is_dynamic: bool = False,
):
    extra_args: Dict[str, Any] = {"eps": 2**-12}
    if is_qat:
        if is_dynamic:
            act_observer_or_fake_quant_ctr = FakeQuantize
            dynamic_quant_observer = MovingAverageMinMaxObserver.with_args(
                averaging_constant=1
            )
            extra_args["observer"] = dynamic_quant_observer
        else:
            act_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize  # type: ignore[assignment]
    else:
        if is_dynamic:
            act_observer_or_fake_quant_ctr = PlaceholderObserver  # type: ignore[assignment]
        else:
            act_observer_or_fake_quant_ctr = HistogramObserver  # type: ignore[assignment]

    # Copy from x86 default qconfig from torch/ao/quantization/qconfig.py
    act_quantization_spec = BFPQuantizationSpec(
        dtype=default_format,
        quant_min=0,
        quant_max=255,  # reduce_range=False
        qscheme=torch.per_tensor_affine,
        is_dynamic=is_dynamic,
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
            **extra_args
        ),
    )

    weight_observer_or_fake_quant_ctr: _ObserverConstructor = (
        FusedMovingAvgObsFakeQuantize if is_qat else PerChannelMinMaxObserver
    )

    if is_qat:
        # Only support per channel quant for now
        extra_args["observer"] = MovingAveragePerChannelMinMaxObserver  # type: ignore[dict-item]
    weight_quantization_spec = BFPQuantizationSpec(
        dtype=default_format,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,  # 0 corresponding to weight shape = (oc, ic, kh, kw) of conv
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(
            **extra_args
        ),
    )
    bias_quantization_spec = None  # will use placeholder observer by default
    quantization_config = BFPQuantizationConfig(
        act_quantization_spec,
        act_quantization_spec,
        weight_quantization_spec,
        bias_quantization_spec,
        is_qat,
    )
    return quantization_config


def _get_supported_config_and_operators() -> List[BFPOperatorConfig]:
    return _get_supported_x86_inductor_config_and_operators()


class X86InductorQuantizer(BFPQuantizer):
    supported_config_and_operators = _get_supported_config_and_operators()
    module_function_to_aten_operator_type = _map_module_function_to_aten_operator_type()

    def __init__(self):
        super().__init__()
        self.global_config: QuantizationConfig = None  # type: ignore[assignment]
        self.operator_type_qconfig: Dict[
            torch._ops.OpOverloadPacket, Optional[BFPQuantizationConfig]
        ] = {}

    @classmethod
    def get_supported_quantization_configs(cls) -> List[BFPQuantizationConfig]:
        op_configs: Set[BFPQuantizationConfig] = {
            spec for spec, _ in cls.supported_config_and_operators
        }
        return list(op_configs)

    @classmethod
    def get_supported_operator_for_quantization_config(
        cls, quantization_config: Optional[BFPQuantizationConfig]
    ) -> List[OperatorPatternType]:
        if quantization_config is None:
            all_ops = []
            for _, ops in cls.supported_config_and_operators:
                all_ops.extend(ops)
            return all_ops

        for config, ops in cls.supported_config_and_operators:
            if config == quantization_config:
                return ops
        return []

    def set_global(self, quantization_config: BFPQuantizationConfig):
        self.global_config = quantization_config
        return self

    def get_global_quantization_config(self):
        if not isinstance(self.global_config, BFPQuantizationConfig):
            warnings.warn(
                "The global_config for X86InductorQuantizer is currently invalid. \
                Please ensure that you use set_global to establish the global quantization configuration."
            )
        return self.global_config

    def set_function_type_qconfig(
        self,
        function_type: Callable,
        quantization_config: Optional[BFPQuantizationConfig],
    ) -> "X86InductorQuantizer":
        if function_type in X86InductorQuantizer.module_function_to_aten_operator_type:
            self._set_aten_operator_qconfig(
                X86InductorQuantizer.module_function_to_aten_operator_type[
                    function_type
                ],
                quantization_config,
            )
        else:
            warnings.warn(
                f"function: Unable to customize quantization config for {function_type} by X86InductorQuantizer."
            )
        return self

    def set_module_type_qconfig(
        self,
        module_type: torch.nn.Module,
        quantization_config: Optional[BFPQuantizationConfig],
    ) -> "X86InductorQuantizer":
        if module_type in X86InductorQuantizer.module_function_to_aten_operator_type:
            self._set_aten_operator_qconfig(
                X86InductorQuantizer.module_function_to_aten_operator_type[module_type],
                quantization_config,
            )
        else:
            warnings.warn(
                f"Module: Unable to customize quantization config for {module_type} by X86InductorQuantizer."
            )
        return self

    def _set_aten_operator_qconfig(
        self,
        operator_type: torch._ops.OpOverloadPacket,
        quantization_config: Optional[BFPQuantizationConfig],
    ) -> "X86InductorQuantizer":
        if operator_type in quantizable_ops:
            self.operator_type_qconfig[operator_type] = quantization_config
        else:
            warnings.warn(
                f"operator: Unable to quantize {operator} by X86InductorQuantizer."
            )
        return self

    def _get_aten_operator_qconfig(
        self,
        operator_type: torch._ops.OpOverloadPacket,
    ) -> Optional[BFPQuantizationConfig]:
        if operator_type in self.operator_type_qconfig:
            assert operator_type in quantizable_ops
            return self.operator_type_qconfig[operator_type]
        return self.global_config if operator_type in default_quantizable_ops else None

    def _annotate_conv_node_helper(
        self,
        conv_node: torch.fx.Node,
        annotate_output: bool,
        quantization_config: BFPQuantizationConfig,
    ) -> None:
        """Helper function to annotate the conv node"""
        input_qspec_map = {}
        input_node = conv_node.args[0]
        assert isinstance(input_node, Node)
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        weight_node = conv_node.args[1]
        assert isinstance(weight_node, Node)
        input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
        bias_node = None if len(conv_node.args) == 2 else conv_node.args[2]
        if isinstance(bias_node, Node):
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
        if annotate_output:
            conv_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )
        else:
            conv_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
            )

    def _annotate_linear_node_helper(
        self,
        linear_node: torch.fx.Node,
        annotate_output: bool,
        quantization_config: BFPQuantizationConfig,
    ) -> None:
        """Helper function to annotate the linear node"""
        input_qspec_map = {}
        assert linear_node.target in (torch.ops.aten.linear.default,)
        has_bias = len(linear_node.args) == 3
        input_index = 0
        weight_index = 1
        bias_index = 2

        input_node = linear_node.args[input_index]
        assert isinstance(input_node, Node)
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)

        weight_node = linear_node.args[weight_index]
        assert isinstance(weight_node, Node)
        input_qspec_map[weight_node] = get_weight_qspec(quantization_config)

        bias_node = linear_node.args[bias_index] if has_bias else None
        if isinstance(bias_node, Node):
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)

        if annotate_output:
            linear_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )
        else:
            linear_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                input_qspec_map=input_qspec_map, _annotated=True
            )

    def _get_output_nodes_of_partitions(
        self,
        partition_list: List[SourcePartition],
    ) -> List[torch.fx.Node]:
        """Helper function to get the output node list from partition list"""
        output_node_list = []
        for partition in partition_list:
            if len(partition.output_nodes) > 1:
                raise ValueError("Input partition has more than one output node")
            output_node = partition.output_nodes[0]
            assert isinstance(output_node, Node)
            output_node_list.append(output_node)
        if len(output_node_list) != len(partition_list):
            raise ValueError(
                "length of output_node_list should equal to length of partition_list"
            )
        return output_node_list

    def _get_input_idx_for_binary_node(
        self,
        conv_gemm_node: torch.fx.Node,
        binary_node: torch.fx.Node,
    ):
        """Helper function to check conv_gemm and extra input node index
        for binary node fused with conv_gemm.
        """
        conv_gemm_node_idx = None
        extra_input_node_idx = None
        if (binary_node.args[0].op == "call_function") and (  # type: ignore[union-attr]
            binary_node.args[0] == conv_gemm_node
        ):
            conv_gemm_node_idx = 0
            extra_input_node_idx = 1
        elif (binary_node.args[1].op == "call_function") and (  # type: ignore[union-attr]
            binary_node.args[1] == conv_gemm_node
        ):
            conv_gemm_node_idx = 1
            extra_input_node_idx = 0
        extra_input_node = binary_node.args[extra_input_node_idx]  # type: ignore[index]
        assert isinstance(extra_input_node, Node)
        return conv_gemm_node_idx, extra_input_node_idx

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """just handling global spec for now"""
        if self.global_config and self.global_config.input_activation.is_dynamic:  # type: ignore[union-attr]
            model = self._annotate_for_dynamic_quantization_config(model)
        else:
            model = self._annotate_for_static_quantization_config(model)
        return model

    def _annotate_for_static_quantization_config(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        r"""
        High-level description of quantization recipe for X86 Inductor Backend:
        Step 1: Apply quantization recipe for fusion patterns of conv/linear to enable int8 data type actively.
        Step 2: Propagate quantization annotation for patterns besides conv/linear. Go through the pattern in model
        from start to the end. If a pattern supports computation with int8 data type and inputs connected to
        quantized patterns, annotate its inputs as quantized pattern.
        Step 3: Since in step 2, we only annotate the inputs of quantized pattern. For some quantized patterns,
        such as maxpool2d, which only supports output with int8 data type when the input is with int8 data type,
        we need to annotate the output of this pattern.
        """

        # Step1: Recipe of fusion patterns like conv/linear.
        self._annotate_conv2d_fusion_pattern(model)
        self._annotate_linear_fusion_pattern(model)
        self._annotate_matmul(model)

        # Step2: Recipe to propagate annotation for patterns beside conv/linear.
        # Go through all the nodes from start to end.
        # Recipe refer to https://github.com/intel/intel-extension-for-pytorch/blob/
        # 90d19323d96afc53fcc22ba5a7bb3fb07fdd6c1c/intel_extension_for_pytorch/quantization/_recipe.py#L538
        for node in model.graph.nodes:
            self._annotate_propagation_quantizable_pattern(node)

        # Step3: For quantizable ops, such as maxpool2d, we need to quantize its output if it is quantized
        # in inputs. So, we can fuse dq-operator-q into a quantized op.
        # Refer to https://github.com/intel/intel-extension-for-pytorch/blob/
        # 90d19323d96afc53fcc22ba5a7bb3fb07fdd6c1c/intel_extension_for_pytorch/quantization/_recipe.py#L487
        for node in model.graph.nodes:
            self._annotate_output_for_int8_in_int8_out_pattern(node)

        return model

    def _annotate_for_dynamic_quantization_config(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        self._annotate_linear_fusion_pattern(model)
        return model

    def _annotate_qat_conv2d_fusion_pattern(
        self, model: torch.fx.GraphModule, config: BFPQuantizationConfig
    ):
        # Annotate QAT Specific patterns
        self._annotate_qat_conv2d_bn_binary_unary(model, config)
        self._annotate_qat_conv2d_bn_binary(model, config)
        self._annotate_qat_conv2d_bn_unary(model, config)
        self._annotate_qat_conv2d_bn(model, config)

    def _annotate_qat_conv2d_bn_binary_unary(
        self, gm: torch.fx.GraphModule, quantization_config: BFPQuantizationConfig
    ) -> None:
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d, operator.add, torch.nn.ReLU]
        )
        for fused_partition in fused_partitions:
            (
                conv_partition,
                bn_partition,
                binary_partition,
                unary_partition,
            ) = fused_partition

            (
                conv_node,
                bn_output_node,
                binary_node,
                unary_node,
            ) = self._get_output_nodes_of_partitions(
                [conv_partition, bn_partition, binary_partition, unary_partition]
            )
            if len(bn_output_node.users) != 1:
                # Conv BN pattern should only has 1 user.
                continue
            (
                bn_output_node_idx,
                extra_input_node_idx,
            ) = self._get_input_idx_for_binary_node(bn_output_node, binary_node)
            if (bn_output_node_idx is None) or (extra_input_node_idx is None):
                continue
            if bn_output_node != binary_node.args[bn_output_node_idx]:
                raise ValueError(f"{bn_output_node} doesn't match input of binary node")
            extra_input_node = binary_node.args[extra_input_node_idx]

            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                continue

            if _is_annotated([unary_node, binary_node, bn_output_node, conv_node]):
                continue

            self._annotate_conv_node_helper(conv_node, False, quantization_config)

            binary_node_input_qspec_map = {}
            binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(
                quantization_config
            )
            binary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                input_qspec_map=binary_node_input_qspec_map,
                _annotated=True,
            )
            unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                # TODO<leslie> Remove the annotate of output in QAT when qat util support pattern matcher.
                output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )
            nodes_to_mark_annotated = list(conv_partition.nodes)
            nodes_to_mark_annotated.extend(list(bn_partition.nodes))
            nodes_to_mark_annotated.extend(list(binary_partition.nodes))
            nodes_to_mark_annotated.extend(list(unary_partition.nodes))
            _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_qat_conv2d_bn_binary(
        self, gm: torch.fx.GraphModule, quantization_config: BFPQuantizationConfig
    ) -> None:
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d, operator.add]
        )
        for fused_partition in fused_partitions:
            conv_partition, bn_partition, binary_partition = fused_partition
            (
                conv_node,
                bn_output_node,
                binary_node,
            ) = self._get_output_nodes_of_partitions(
                [conv_partition, bn_partition, binary_partition]
            )
            if len(bn_output_node.users) != 1:
                # Conv BN pattern should only has 1 user.
                continue
            (
                bn_output_node_idx,
                extra_input_node_idx,
            ) = self._get_input_idx_for_binary_node(bn_output_node, binary_node)
            if (bn_output_node_idx is None) or (extra_input_node_idx is None):
                continue
            if bn_output_node != binary_node.args[bn_output_node_idx]:
                raise ValueError(f"{bn_output_node} doesn't match input of binary node")

            extra_input_node = binary_node.args[extra_input_node_idx]

            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                continue

            if _is_annotated([binary_node, bn_output_node, conv_node]):
                continue

            self._annotate_conv_node_helper(conv_node, False, quantization_config)

            binary_node_input_qspec_map = {}
            binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(
                quantization_config
            )
            binary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                input_qspec_map=binary_node_input_qspec_map,
                # TODO<leslie> Remove the annotate of output in QAT when qat util support pattern matcher.
                output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )
            nodes_to_mark_annotated = list(conv_partition.nodes)
            nodes_to_mark_annotated.extend(list(bn_partition.nodes))
            nodes_to_mark_annotated.extend(list(binary_partition.nodes))
            _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_qat_conv2d_bn_unary(
        self, gm: torch.fx.GraphModule, quantization_config: BFPQuantizationConfig
    ) -> None:
        fused_partitions = []
        unary_patterns = [
            [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU],
            [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Hardtanh],
            [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Hardswish],
            [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU6],
            [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.SiLU],
        ]
        for unary_pattern in unary_patterns:
            partitions = find_sequential_partitions(gm, unary_pattern)
            if partitions:
                # Extend the fused_partitions if partitions is not empty
                fused_partitions.extend(partitions)

        for fused_partition in fused_partitions:
            conv_partition, bn_partition, unary_partition = fused_partition
            (
                conv_node,
                bn_output_node,
                unary_node,
            ) = self._get_output_nodes_of_partitions(
                [conv_partition, bn_partition, unary_partition]
            )

            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                continue

            if _is_annotated([unary_node, bn_output_node, conv_node]):
                continue

            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                # TODO<leslie> Remove the annotate of output in QAT when qat util support pattern matcher.
                output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )
            nodes_to_mark_annotated = list(conv_partition.nodes)
            nodes_to_mark_annotated.extend(list(bn_partition.nodes))
            nodes_to_mark_annotated.extend(list(unary_partition.nodes))
            _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_qat_conv2d_bn(
        self, gm: torch.fx.GraphModule, quantization_config: BFPQuantizationConfig
    ) -> None:
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, torch.nn.BatchNorm2d]
        )
        for fused_partition in fused_partitions:
            conv_partition, bn_partition = fused_partition
            conv_node, bn_output_node = self._get_output_nodes_of_partitions(
                [conv_partition, bn_partition]
            )

            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                continue

            if _is_annotated([bn_output_node, conv_node]):
                continue

            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            bn_output_node.meta[
                QUANT_ANNOTATION_KEY
            ] = _X86InductorQuantizationAnnotation(
                # TODO<leslie> Remove the annotate of output in QAT when qat util support pattern matcher.
                output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )
            nodes_to_mark_annotated = list(conv_partition.nodes)
            nodes_to_mark_annotated.extend(list(bn_partition.nodes))
            _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def _annotate_conv2d_fusion_pattern(self, model: torch.fx.GraphModule):
        if config := self._get_aten_operator_qconfig(torch.ops.aten.conv2d.default):
            if config.is_qat:
                # Annotate QAT specific pattern: mainly due to BN not folded in prepare_qat
                self._annotate_qat_conv2d_fusion_pattern(model, config)
            self._annotate_conv2d_binary_unary(model, config)
            self._annotate_conv2d_binary(model, config)
            self._annotate_conv2d_unary(model, config)
            self._annotate_conv2d(model, config)

    def _annotate_linear_fusion_pattern(self, model: torch.fx.GraphModule):
        if config := self._get_aten_operator_qconfig(torch.ops.aten.linear.default):
            if config.input_activation and not config.input_activation.is_dynamic:
                # <TODO> Weiwen: Dynamic Quant of linear unary will be supported in next step
                self._annotate_linear_binary_unary(model, config)
                self._annotate_linear_unary(model, config)
            self._annotate_linear(model, config)

    def _annotate_matmul(self, model: torch.fx.GraphModule):
        if config := self._get_aten_operator_qconfig(torch.ops.aten.matmul.default):
            for node in model.graph.nodes:
                if node.target == torch.ops.aten.matmul.default and not _is_annotated(
                    [node]
                ):
                    input_qspec_map = {}
                    matmul_node = node
                    for input_node in matmul_node.args:
                        input_qspec_map[input_node] = get_input_act_qspec(config)
                    matmul_node.meta[
                        QUANT_ANNOTATION_KEY
                    ] = _X86InductorQuantizationAnnotation(
                        input_qspec_map=input_qspec_map,
                        _annotated=True,
                        _is_output_of_quantized_pattern=True,
                    )

    def _annotate_conv2d_binary_unary(
        self, gm: torch.fx.GraphModule, quantization_config: BFPQuantizationConfig
    ) -> None:
        # Conv2d + add + unary op
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, operator.add, torch.nn.ReLU]
        )
        for fused_partition in fused_partitions:
            conv_partition, binary_partition, unary_partition = fused_partition
            conv_node, binary_node, unary_node = self._get_output_nodes_of_partitions(
                [conv_partition, binary_partition, unary_partition]
            )
            if len(conv_node.users) != 1:
                # Conv Node should only has 1 user node
                continue
            conv_node_idx, extra_input_node_idx = self._get_input_idx_for_binary_node(
                conv_node, binary_node
            )
            if (conv_node_idx is None) or (extra_input_node_idx is None):
                continue
            if conv_node != binary_node.args[conv_node_idx]:
                raise ValueError(f"{conv_node} doesn't match input of binary node")
            extra_input_node = binary_node.args[extra_input_node_idx]
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                # No conv node found to be fused with add
                continue
            if _is_annotated([unary_node, binary_node, conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            binary_node_input_qspec_map = {}
            binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(
                quantization_config
            )
            binary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                input_qspec_map=binary_node_input_qspec_map,
                _annotated=True,
            )
            unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )

    def _annotate_conv2d_binary(
        self, gm: torch.fx.GraphModule, quantization_config: BFPQuantizationConfig
    ) -> None:
        # Conv2d + add
        fused_partitions = find_sequential_partitions(
            gm, [torch.nn.Conv2d, operator.add]
        )
        for fused_partition in fused_partitions:
            conv_partition, binary_partition = fused_partition
            conv_node, binary_node = self._get_output_nodes_of_partitions(
                [conv_partition, binary_partition]
            )
            if len(conv_node.users) != 1:
                # Conv Node should only has 1 user node
                continue
            conv_node_idx, extra_input_node_idx = self._get_input_idx_for_binary_node(
                conv_node, binary_node
            )
            if (conv_node_idx is None) or (extra_input_node_idx is None):
                continue
            if conv_node != binary_node.args[conv_node_idx]:
                raise ValueError(f"{conv_node} doesn't match input of binary node")
            extra_input_node = binary_node.args[extra_input_node_idx]
            assert isinstance(conv_node, Node)
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                # No conv node found to be fused with add
                continue
            if _is_annotated([binary_node, conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            binary_node_input_qspec_map = {}
            binary_node_input_qspec_map[extra_input_node] = get_input_act_qspec(
                quantization_config
            )
            binary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                input_qspec_map=binary_node_input_qspec_map,
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )

    def _annotate_conv2d_unary(
        self, gm: torch.fx.GraphModule, quantization_config: BFPQuantizationConfig
    ) -> None:
        fused_partitions = []
        unary_patterns = [
            [torch.nn.Conv2d, torch.nn.ReLU],
            [torch.nn.Conv2d, torch.nn.Hardtanh],
            [torch.nn.Conv2d, torch.nn.Hardswish],
            [torch.nn.Conv2d, torch.nn.ReLU6],
            [torch.nn.Conv2d, torch.nn.SiLU],
        ]
        for unary_pattern in unary_patterns:
            partitions = find_sequential_partitions(gm, unary_pattern)
            if partitions:
                # Extend the fused_partitions if partitions is not empty
                fused_partitions.extend(partitions)

        for fused_partition in fused_partitions:
            conv_partition, unary_partition = fused_partition
            conv_node, unary_node = self._get_output_nodes_of_partitions(
                [conv_partition, unary_partition]
            )
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                continue
            if _is_annotated([unary_node, conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, False, quantization_config)
            unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )

    def _annotate_conv2d(
        self, gm: torch.fx.GraphModule, quantization_config: BFPQuantizationConfig
    ) -> None:
        conv_partitions = get_source_partitions(
            gm.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d]
        )
        conv_partitions = list(itertools.chain.from_iterable(conv_partitions.values()))
        for conv_partition in conv_partitions:
            if len(conv_partition.output_nodes) > 1:
                raise ValueError("conv partition has more than one output node")
            conv_node = conv_partition.output_nodes[0]
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.conv2d.default
            ):
                raise ValueError(f"{conv_node} is not an aten conv2d operator")
            # skip annotation if it is already annotated
            if _is_annotated([conv_node]):
                continue
            self._annotate_conv_node_helper(conv_node, True, quantization_config)

    def _annotate_maxpool2d(
        self, node: Node, quantization_config: BFPQuantizationConfig
    ) -> None:
        if node.target is not torch.ops.aten.max_pool2d.default:
            return
        maxpool_node = node
        if _is_any_annotated(
            [
                maxpool_node,
            ]
        ):
            return
        input_node = maxpool_node.args[0]
        assert isinstance(input_node, Node)
        input_qspec_map = {}
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        maxpool_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
            _is_output_of_quantized_pattern=True,
        )

    def _annotate_cat(
        self, node: Node, quantization_config: BFPQuantizationConfig
    ) -> None:
        cat_node = node
        input_nodes = cat_node.args[0]
        assert isinstance(input_nodes, Sequence)
        first_input_node = input_nodes[0]
        input_qspec_map = {}
        assert isinstance(first_input_node, Node)
        assert isinstance(cat_node, Node)
        input_qspec_map[first_input_node] = get_input_act_qspec(quantization_config)
        share_qparams_with_input_act0_qspec = SharedBFPQuantizationSpec(
            (first_input_node, cat_node)
        )

        for input_node in input_nodes[1:]:
            if input_node not in input_qspec_map:
                # There has the case of cat same nodes: torch.cat([input0, input0], 1)
                assert isinstance(input_node, Node)
                input_qspec_map[input_node] = share_qparams_with_input_act0_qspec

        cat_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
            _is_output_of_quantized_pattern=True,
        )

    def _annotate_propagation_quantizable_pattern(self, node: Node) -> None:
        # Propagate annotation to quantizable patterns.
        if (
            (node.target in propagation_quantizable_ops)
            and (not _is_any_annotated([node]))
            and (node.op == "call_function")
            and (quantization_config := self._get_aten_operator_qconfig(node.target))  # type: ignore[arg-type]
        ):

            def is_all_inputs_connected_to_quantized_op(input_nodes):
                # Ensure all the inputs connect to fusion pattern or quantized node
                for input_node in input_nodes:
                    if not _is_quantized_op_pt2e(input_node):
                        return False
                return True

            if node.target is torch.ops.aten.max_pool2d.default:
                # Recipe of maxpool2d: check input arg[0] of maxpool2d is quantized or not
                input_nodes_to_check = [node.all_input_nodes[0]]
                if not is_all_inputs_connected_to_quantized_op(input_nodes_to_check):
                    return
                self._annotate_maxpool2d(node, quantization_config)
                return
            elif node.target is torch.ops.aten.cat.default:
                input_nodes_to_check = node.all_input_nodes
                if not is_all_inputs_connected_to_quantized_op(input_nodes_to_check):
                    return
                self._annotate_cat(node, quantization_config)
            else:
                input_node = node.all_input_nodes[0]
                if not is_all_inputs_connected_to_quantized_op(
                    [
                        input_node,
                    ]
                ):
                    return
                input_qspec_map = {}
                input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
                node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                    input_qspec_map=input_qspec_map,
                    _annotated=True,
                    _is_output_of_quantized_pattern=True,
                )
        return

    def _annotate_output_share_observer_as_input(
        self, input_node: Node, source_node: Node
    ):
        source_node_quantization_annotation = (
            source_node.meta[QUANT_ANNOTATION_KEY]
            if QUANT_ANNOTATION_KEY in source_node.meta
            else None
        )
        if (
            source_node_quantization_annotation
            and source_node_quantization_annotation._is_output_of_quantized_pattern
        ):
            edge_or_node = (input_node, source_node)
            source_node_quantization_annotation.output_qspec = SharedBFPQuantizationSpec(
                edge_or_node
            )
        return

    def _annotate_output_for_int8_in_int8_out_pattern(self, node: Node) -> None:
        r"""
        Check and insert observer at output of node in int8_in_int8_out_ops if needed.
        Recipe refers to https://github.com/intel/intel-extension-for-pytorch/blob/
        90d19323d96afc53fcc22ba5a7bb3fb07fdd6c1c/intel_extension_for_pytorch/quantization/_utils.py#L495
        """
        edge_or_node: Tuple[Node, Node]
        if (
            (node.target in int8_in_int8_out_ops)
            and (_is_any_annotated([node]))
            and (quantization_config := self._get_aten_operator_qconfig(node.target))  # type: ignore[arg-type]
        ):
            if node.target == torch.ops.aten.max_pool2d.default:
                maxpool_node = node
                if not _is_all_annotated(
                    [
                        maxpool_node,
                    ]
                ):
                    return
                # Get the quantization_annotation from getitem_node
                maxpool_node_quantization_annotation = (
                    maxpool_node.meta[QUANT_ANNOTATION_KEY]
                    if QUANT_ANNOTATION_KEY in maxpool_node.meta
                    else None
                )
                if (
                    maxpool_node_quantization_annotation
                    and maxpool_node_quantization_annotation._is_output_of_quantized_pattern
                ):
                    # Annotate the output_qspec of getitem_node
                    input_act = maxpool_node.args[0]
                    assert isinstance(input_act, Node)
                    assert isinstance(maxpool_node, Node)
                    edge_or_node = (input_act, maxpool_node)
                    maxpool_node_quantization_annotation.output_qspec = (
                        SharedQuantizationSpec(edge_or_node)
                    )
            else:
                input_node = node.all_input_nodes[0]
                self._annotate_output_share_observer_as_input(input_node, node)
        return

    def _annotate_linear(
        self, gm: torch.fx.GraphModule, quantization_config: BFPQuantizationConfig
    ) -> None:
        linear_partitions = get_source_partitions(
            gm.graph, [torch.nn.Linear, torch.nn.functional.linear]
        )
        linear_partitions = list(
            itertools.chain.from_iterable(linear_partitions.values())
        )
        for partition in linear_partitions:
            if len(partition.output_nodes) > 1:
                raise ValueError(
                    "Linear partition cannot have more than one output node"
                )
            linear_node = partition.output_nodes[0]
            if linear_node.op != "call_function" or linear_node.target not in (
                torch.ops.aten.linear.default,
            ):
                raise ValueError(f"{linear_node} is not an aten linear operator")
            # skip annotation if it is already annotated
            if _is_annotated([linear_node]):
                continue
            self._annotate_linear_node_helper(linear_node, True, quantization_config)

    def _annotate_linear_unary(
        self, gm: torch.fx.GraphModule, quantization_config: BFPQuantizationConfig
    ) -> None:
        postop_list = [
            torch.nn.ReLU,
            torch.nn.LeakyReLU,
            torch.nn.Tanh,
            torch.nn.GELU,
        ]
        fused_partitions: List[tuple] = []
        for postop in postop_list:
            fused_partitions = fused_partitions + find_sequential_partitions(
                gm, [torch.nn.Linear, postop]
            )
        for fused_partition in fused_partitions:
            linear_partition, unary_partition = fused_partition
            linear_node, unary_node = self._get_output_nodes_of_partitions(
                [linear_partition, unary_partition]
            )
            if linear_node.op != "call_function" or linear_node.target not in (
                torch.ops.aten.linear.default,
            ):
                continue
            if _is_annotated([unary_node, linear_node]):
                continue
            self._annotate_linear_node_helper(linear_node, False, quantization_config)
            unary_node.meta[QUANT_ANNOTATION_KEY] = _X86InductorQuantizationAnnotation(
                _annotated=True,
                _is_output_of_quantized_pattern=True,
            )

    def _annotate_linear_binary_unary(
        self,
        gm: torch.fx.GraphModule,
        quantization_config: BFPQuantizationConfig,
    ) -> None:
        # linear + binary_op + (optional) unary op
        binary_op_list = [operator.add]
        unary_op_list = [torch.nn.ReLU, None]
        combinations = itertools.product(binary_op_list, unary_op_list)
        for binary_op, unary_op in combinations:
            has_unary = unary_op is not None
            seq_partition = [torch.nn.Linear, binary_op]
            if has_unary:
                seq_partition.append(unary_op)
            fused_partitions = find_sequential_partitions(gm, seq_partition)
            for fused_partition in fused_partitions:
                unary_partition, unary_node = None, None
                if has_unary:
                    (
                        linear_partition,
                        binary_partition,
                        unary_partition,
                    ) = fused_partition
                    (
                        linear_node,
                        binary_node,
                        unary_node,
                    ) = self._get_output_nodes_of_partitions(
                        [linear_partition, binary_partition, unary_partition]
                    )
                else:
                    linear_partition, binary_partition = fused_partition
                    linear_node, binary_node = self._get_output_nodes_of_partitions(
                        [linear_partition, binary_partition]
                    )
                if len(linear_node.users) != 1:
                    # Linear Node should only has 1 user node
                    continue
                (
                    linear_node_idx,
                    extra_input_node_idx,
                ) = self._get_input_idx_for_binary_node(linear_node, binary_node)
                if (linear_node_idx is None) or (extra_input_node_idx is None):
                    continue
                if linear_node != binary_node.args[linear_node_idx]:
                    raise ValueError(
                        f"{linear_node} doesn't match input of binary node"
                    )
                assert isinstance(linear_node, Node)
                if (
                    linear_node.op != "call_function"
                    or linear_node.target != torch.ops.aten.linear.default
                ):
                    # No linear node found to be fused with add
                    continue
                node_list = (
                    [binary_node, linear_node]
                    if unary_node is None
                    else [unary_node, binary_node, linear_node]
                )
                if _is_annotated(node_list):
                    continue
                self._annotate_linear_node_helper(
                    linear_node, False, quantization_config
                )
                # We don't insert q-dq before the binary input node due to accuracy issues
                binary_node.meta[
                    QUANT_ANNOTATION_KEY
                ] = _X86InductorQuantizationAnnotation(
                    input_qspec_map={},
                    _annotated=True,
                    _is_output_of_quantized_pattern=(not has_unary),
                )
                if unary_node is not None:
                    unary_node.meta[
                        QUANT_ANNOTATION_KEY
                    ] = _X86InductorQuantizationAnnotation(
                        _annotated=True,
                        _is_output_of_quantized_pattern=True,
                    )

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[BFPOperatorConfig]:
        return cls.supported_config_and_operators