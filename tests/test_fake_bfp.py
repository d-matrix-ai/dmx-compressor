import copy
import itertools
import operator
from typing import Callable, Dict, List, Optional, Set, Any
from dmx.compressor.numerical.format import Format, Same

from dmx.compressor.numerical.cast import CastTo
from dmx.compressor.numerical.observer import MinMaxObserver

import sys

import transformers

sys.path.insert(0, '../src')

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2", )
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
# model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

default_format = Format.from_shorthand("BFP[8|8]{64}(SN)")

import torch
import torch._dynamo as torchdynamo

from pt2bfp.utils import(
    _annotate_input_qspec_map,
    _annotate_output_qspec,
    get_input_act_qspec,
    get_output_act_qspec,
    get_bias_qspec,
    get_weight_qspec,
    BFPOperatorConfig,
    BFPQuantizationConfig,
    BFPQuantizationSpec)

from pt2bfp.quantizer import (
    BFPQuantizer,
    BFPQuantizationAnnotation,
    SharedBFPQuantizationSpec
)

from torch.fx import Node

from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

from pt2bfp.observer import (
    HistogramObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)
from pt2bfp.qconfig import _ObserverConstructor
from pt2bfp.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
)

from pt2bfp.fake_quantize import default_fake_quant, default_weight_fake_quant

def _mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if node is not None:
            if "quantization_annotation" not in node.meta:
                node.meta["quantization_annotation"] = BFPQuantizationAnnotation()
            node.meta["quantization_annotation"]._annotated = True

def _is_annotated(nodes: List[Node]):
    annotated = False
    for node in nodes:
        annotated = annotated or (
            "quantization_annotation" in node.meta
            and node.meta["quantization_annotation"]._annotated
        )
    return annotated

class BackendQuantizer(BFPQuantizer):

    def __init__(self):
        super().__init__()
        self.global_config: QuantizationConfig = None  # type: ignore[assignment]
        self.operator_type_config: Dict[str, Optional[BFPQuantizationConfig]] = {}

    def set_global(self, quantization_config: BFPQuantizationConfig):
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
        self, model: torch.fx.GraphModule, config: BFPQuantizationConfig
    ) -> torch.fx.GraphModule:
        self._annotate_linear(model, config)
        self._annotate_conv(model, config)
        return model

    def _annotate_conv(
        self, gm: torch.fx.GraphModule, quantization_config: BFPQuantizationConfig
    ) -> None:
        conv_partitions = get_source_partitions(
            gm.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d, torch.nn.Conv1d, torch.nn.functional.conv1d, transformers.pytorch_utils.Conv1D]
        )
        conv_partitions = list(itertools.chain(*conv_partitions.values()))
        for conv_partition in conv_partitions:
            if len(conv_partition.output_nodes) > 1:
                raise ValueError("conv partition has more than one output node")
            conv_node = conv_partition.output_nodes[0]
            if (
                conv_node.op != "call_function"
                or conv_node.target != torch.ops.aten.convolution.default
            ):
                raise ValueError(f"{conv_node} is not an aten conv2d operator")
            # skip annotation if it is already annotated
            if _is_annotated([conv_node]):
                continue

            input_qspec_map = {}
            input_act = conv_node.args[0]
            assert isinstance(input_act, Node)
            input_qspec_map[input_act] = get_input_act_qspec(quantization_config)

            weight = conv_node.args[1]
            assert isinstance(weight, Node)
            input_qspec_map[weight] = get_weight_qspec(quantization_config)

            bias = conv_node.args[2]
            if isinstance(bias, Node):
                input_qspec_map[bias] = get_bias_qspec(quantization_config)

            conv_node.meta["quantization_annotation"] = BFPQuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )

    def _annotate_linear(
        self, gm: torch.fx.GraphModule, quantization_config: BFPQuantizationConfig
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
    def get_supported_operators(cls) -> List[BFPOperatorConfig]:
        return []

def get_symmetric_quantization_config(dtype = default_format):
    '''
    When using CastTo have to comment out the quant_mi and quant_max assertions in the pytorch files
    File "/path/to/python/site-packages/torch/ao/quantization/fake_quantize.py", line 166, in __init__
    assert torch.iinfo(dtype).min <= quant_min, 'quant_min out of bound'
    '''
    act_quantization_spec = BFPQuantizationSpec(
        dtype=torch.float,
        qscheme=torch.per_tensor_affine,
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=CastTo.with_args(format = default_format, observer = MinMaxObserver),
    )

    weight_quantization_spec = BFPQuantizationSpec(
        dtype=torch.float,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=CastTo.with_args(format = default_format, observer = MinMaxObserver),
    )

    bias_observer_or_fake_quant_ctr: _ObserverConstructor = PlaceholderObserver
    # bias_quantization_spec = BFPQuantizationSpec(
    #     dtype=torch.float,
    #     observer_or_fake_quant_ctr=bias_observer_or_fake_quant_ctr
    # )
    bias_quantization_spec = BFPQuantizationSpec(
        dtype=torch.float,
        qscheme=torch.per_tensor_symmetric,
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=CastTo.with_args(format = default_format, observer = MinMaxObserver),
    )

    quantization_config = BFPQuantizationConfig(
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

def distilgpt2_generate_text(text):
    input = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(input, max_length=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    # example_inputs = (torch.randn(1, 3, 224, 224),)
    # m = torchvision.models.resnet18().eval()
    torch.manual_seed(0)
    # print([module for module in model.modules()])
    tokenizer.pad_token = tokenizer.eos_token
    input = tokenizer.encode("What is the capital of France?", return_tensors="pt", max_length = 384, padding="max_length")
    example_inputs = (torch.randn(1, 1024),)
    # m = M().eval()
    m = model.eval()
    m_base = copy.deepcopy(m)
    # program capture
    m, guards = torchdynamo.export(
        m,
        copy.deepcopy(input),
        # copy.deepcopy(*example_inputs),
        aten_graph=True,
    )    
    quantizer = BackendQuantizer()
    operator_config = get_symmetric_quantization_config(default_format)
    quantizer.set_global(operator_config)
    # Note: ``prepare_pt2e_quantizer`` will be updated to ``prepare_pt2e`` soon
    # before_prepare_result = m(input)
    m = prepare_pt2e(m, quantizer)
    # after_prepare_result = m(input)
    # assert torch.all((before_prepare_result-after_prepare_result) == torch.zeros_like(before_prepare_result))
    m = convert_pt2e(m)
    # import ipdb; ipdb.set_trace()
    out = m(*input)
    # out_base = m_base(*example_inputs)
    # print(torch.equal(out, out_base))
    # print("converted module is: {}".format(m), flush=True)


    

