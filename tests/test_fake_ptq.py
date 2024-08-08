import copy
from typing import Dict, List, Optional, Set, Any

import torch
import torch._dynamo as torchdynamo
from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec)

from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import(
    get_input_act_qspec,
    get_output_act_qspec,
    get_bias_qspec,
    get_weight_qspec,
    OperatorConfig,
    QuantizationConfig,
    QuantizationSpec)
from torch.ao.quantization.quantizer.quantizer import (
    Quantizer,
    QuantizationAnnotation,
    SharedQuantizationSpec
)

from torch.fx import Node

from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from torch.ao.quantization.observer import (
    HistogramObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
)

from torch.ao.quantization.fake_quantize import default_fake_quant, default_weight_fake_quant

def _mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if node is not None:
            if "quantization_annotation" not in node.meta:
                node.meta["quantization_annotation"] = QuantizationAnnotation()
            node.meta["quantization_annotation"]._annotated = True

def _is_annotated(nodes: List[Node]):
    annotated = False
    for node in nodes:
        annotated = annotated or (
            "quantization_annotation" in node.meta
            and node.meta["quantization_annotation"]._annotated
        )
    return annotated

class BackendQuantizer(Quantizer):

    def __init__(self):
        super().__init__()
        self.global_config: QuantizationConfig = None  # type: ignore[assignment]
        self.operator_type_config: Dict[str, Optional[QuantizationConfig]] = {}

    def set_global(self, quantization_config: QuantizationConfig):
        """set global QuantizationConfig used for the backend.
        QuantizationConfig is defined in torch/ao/quantization/_pt2e/quantizer/quantizer.py.
        """
        self.global_config = quantization_config
        return self

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """annotate nodes in the graph with observer or fake quant constructors
        to convey the desired way of quantization.
        """
        global_config = self.global_config
        self.annotate_symmetric_config(model, global_config)

        return model

    def annotate_symmetric_config(
        self, model: torch.fx.GraphModule, config: QuantizationConfig
    ) -> torch.fx.GraphModule:
        self._annotate_linear(model, config)
        return model

    def _annotate_linear(
        self, gm: torch.fx.GraphModule, quantization_config: QuantizationConfig
    ) -> None:
        module_partitions = get_source_partitions(
            gm.graph, [torch.nn.Linear, torch.nn.functional.linear]
        )
        act_qspec = get_input_act_qspec(quantization_config)
        weight_qspec = get_weight_qspec(quantization_config)
        bias_qspec = get_bias_qspec(quantization_config)
        for module_or_fn_type, partitions in module_partitions.items():
            if module_or_fn_type == torch.nn.Linear:
                for p in partitions:
                    act_node = p.input_nodes[0]
                    output_node = p.output_nodes[0]
                    weight_node = None
                    bias_node = None
                    for node in p.params:
                        weight_or_bias = getattr(gm, node.target)  # type: ignore[arg-type]
                        if weight_or_bias.ndim == 2:  # type: ignore[attr-defined]
                            weight_node = node
                        if weight_or_bias.ndim == 1:  # type: ignore[attr-defined]
                            bias_node = node
                    if weight_node is None:
                        raise ValueError("No weight found in Linear pattern")
                    # find use of act node within the matched pattern
                    act_use_node = None
                    for node in p.nodes:
                        if node in act_node.users:  # type: ignore[union-attr]
                            act_use_node = node
                            break
                    if act_use_node is None:
                        raise ValueError(
                            "Could not find an user of act node within matched pattern."
                        )
                    if _is_annotated([act_use_node]) is False:  # type: ignore[list-item]
                        _annotate_input_qspec_map(
                            act_use_node,
                            act_node,
                            act_qspec,
                        )
                    if bias_node and _is_annotated([bias_node]) is False:
                        _annotate_output_qspec(bias_node, bias_qspec)
                    if _is_annotated([weight_node]) is False:  # type: ignore[list-item]
                        _annotate_output_qspec(weight_node, weight_qspec)
                    if _is_annotated([output_node]) is False:
                        _annotate_output_qspec(output_node, act_qspec)
                    nodes_to_mark_annotated = list(p.nodes)
                    _mark_nodes_as_annotated(nodes_to_mark_annotated)

    def validate(self, model: torch.fx.GraphModule) -> None:
        """validate if the annotated graph is supported by the backend"""
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return []

def get_symmetric_quantization_config():
    act_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
        observer_or_fake_quant_ctr=default_fake_quant,
    )

    weight_quantization_spec = QuantizationSpec(
        dtype = torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=default_weight_fake_quant,
    )

    bias_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = PlaceholderObserver
    bias_quantization_spec = QuantizationSpec(
        dtype=torch.float,
        observer_or_fake_quant_ctr=bias_observer_or_fake_quant_ctr
    )
    quantization_config = QuantizationConfig(
        act_quantization_spec,
        act_quantization_spec,
        weight_quantization_spec,
        bias_quantization_spec,
    )
    return quantization_config

class M(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.linear = torch.nn.Linear(1024, 1000)

   def forward(self, x):
      return self.linear(x)


if __name__ == "__main__":
    # example_inputs = (torch.randn(1, 3, 224, 224),)
    # m = torchvision.models.resnet18().eval()
    torch.manual_seed(0)
    example_inputs = (torch.randn(1, 1024),)
    m = M().eval()
    m_copy = copy.deepcopy(m)
    # program capture
    m, guards = torchdynamo.export(
        m,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
    )    
    quantizer = BackendQuantizer()
    operator_config = get_symmetric_quantization_config()
    quantizer.set_global(operator_config)
    # Note: ``prepare_pt2e_quantizer`` will be updated to ``prepare_pt2e`` soon
    m = prepare_pt2e(m, quantizer)
    # import ipdb; ipdb.set_trace()
    after_prepare_result = m(*example_inputs)
    m = convert_pt2e(m)
    print("converted module is: {}".format(m), flush=True)

import ipdb; ipdb.set_trace()
