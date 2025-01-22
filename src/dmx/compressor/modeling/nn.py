from abc import abstractmethod
from typing import Union, List, Optional
from types import SimpleNamespace
from collections import OrderedDict
import torch
from torch import Tensor, Size
import torch.nn.functional as F
from torch.fx import Graph, Node, symbolic_trace
import transformers
import transformers.activations

from dmx.compressor.numerical import NumericalCastMixin, Same, CastTo, CastToDict
from dmx.compressor.sparse import (
    WeightSparseMixin,
    Dense,
    LazySparsify,
)
from dmx.compressor.functional import (
    ApproximationMixin,
    NoApproximation,
)
from dmx.compressor.perf_proxy import PerformanceProxyMixin
from dmx.compressor.layer_reconstruction import LayerReconstructionMixin

import math


class DmxModuleType(type):
    pass


class DmxModule(
    ApproximationMixin,
    PerformanceProxyMixin,
    LayerReconstructionMixin,
    NumericalCastMixin,
    WeightSparseMixin,
    torch.nn.Module,
    metaclass=DmxModuleType,
):
    r"""
    Extended torch.nn.Module for Dmx to support quantization.

    Args:
        *args (Optional[Tuple]): variable length of args
        state_dict_url (Optional[str]): Url for loading the state dicts. Defaults to None.
        **kwargs (Optional[Dict]): variable length of kwargs

    Attributes:
        state_dict_url (str): Url for loading the module state dicts.
    """

    is_compound = False

    def __init__(self, *args, state_dict_url: Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state_dict_url = state_dict_url
        if state_dict_url is not None:
            self.load_state_dict_and_register_url(state_dict_url)

    def configure(self, config) -> None:
        """
        A function that changes the format of the ops and loading state dicts according to the config file.

        Args:
            config (DmxModuleConfig): config file for setting new formats and loading state dicts.

        """
        # numerics transformation
        if "input_formats" in config:
            self.input_casts.set_format(format=config["input_formats"])
        if "output_format" in config:
            self.output_cast.set_format(format=config["output_format"])
        if self.accum_cast is not None and "accum_format" in config:
            self.accum_cast.set_format(format=config["accum_format"])
        if self.weight_cast is not None and "weight_format" in config:
            self.weight_cast.set_format(format=config["weight_format"])
        if self.bias_cast is not None and "bias_format" in config:
            self.bias_cast.set_format(format=config["bias_format"])
        if self.smoothquant is not None and "smoothquant_scale_format" in config:
            self.smoothquant.set_scale_format(format=config["smoothquant_scale_format"])
        # sparsity transformation
        if self.weight_sparsifier is not None and "weight_sparseness" in config:
            self.weight_sparsifier.configure(sparseness=config["weight_sparseness"])
        # custom logic transformation
        if "approximation_function" in config:
            self.approximator.set_function(config["approximation_function"])

        # load state_dict if provided
        if (
            "state_dict_url" in config
            and config["state_dict_url"] != self.state_dict_url
        ):
            self.load_state_dict_and_register_url(config["state_dict_url"])

    transform = configure  # NOTE: to be deprecated

    def load_state_dict_and_register_url(self, url: str) -> None:
        """
        A function that loads state dict from a url and sets url to self.state_dict_url

        Args:
            ulr (str): url for loading the state dict
        """
        self.load_state_dict(torch.hub.load_state_dict_from_url(url))
        self.state_dict_url = url

    def save_state_dict_and_register_url(self, parent_dir: str) -> None:
        """
        A function that saves the current state dict of the module to a url under a specified parent directory

        Args:
            parent_dir (str): parent directory for the url
        """
        from dmx.compressor.utils import save_state_dict_and_return_url

        url = save_state_dict_and_return_url(self, parent_dir)
        self.state_dict_url = url

    def dmx_config(self, freeze=False):
        """
        A function that the DmxModuleConfig object for the module

        Args:
            freeze (bool): if True, both state dict and ops formats would be included in the returned DmxModuleConfig. If False, only state dict will be included.

        Returns:
            A DmxModuleConfig object for the module
        """
        return DmxModuleConfig.from_module(self, freeze)

    def fold_weight_and_bias(self) -> None:
        """
        A function that applies the ops the weights and biases using the corresponding formats.
        """
        with torch.no_grad():
            # bias cast
            if self.bias_cast is not None and not isinstance(self.bias_format, Same):
                self.bias.data = self.bias_cast(self.bias.data)
                self.bias_cast = CastTo(format=Same())
            # weight sparsifier
            if self.weight_sparsifier is not None and not isinstance(
                self.weight_sparseness, Dense
            ):
                self.weight.data = self.effective_weight
                self.weight_sparsifier = LazySparsify(sparseness=Dense())
            # weight smoothquant
            if (
                self.smoothquant is not None
                and self.smoothquant.fused_to_weight[0] == 0
            ):
                self.smoothquant.fuse_to_weight(self.weight)
            # weight cast
            if self.weight_cast is not None and not isinstance(self.weight_cast, Same):
                self.weight.data = self.weight_cast(self.weight.data)
                self.weight_cast = CastTo(format=Same())

    @property
    def weight_hypernet(self):
        """
        Returns a function that processes weight according to the ops format of the module
        """

        def _weight_hypernet(_w):
            if self.weight_sparsifier is not None:
                _w = self.weight_sparsifier(_w)
            if (
                self.smoothquant is not None
                and self.smoothquant.fused_to_weight[0] == 0
            ):
                _w = self.smoothquant.scale_weight(_w)
            if self.weight_cast is not None:
                _w = self.weight_cast(_w)
            return _w

        return _weight_hypernet

    @property
    def _weight(self):
        """
        Returns the quantized weights of the module
        """
        return self.weight_hypernet(self.weight)

    @property
    def weight_scale(self):
        """
        Returns the quantization scale of the weight matrix
        """
        return self.weight_cast.scale.to(self.weight.device)

    @property
    def weight_zero_point(self):
        """
        Returns the quantization zero_point of the weight matrix
        """
        return self.weight_cast.zero_point.to(self.weight.device)

    @property
    def _bias(self):
        """
        Returns the quantized bias of the module
        """

        return self.bias_cast(self.bias) if self.bias_cast is not None else None

    def forward(self, input: Tensor, *args, **kwags) -> Tensor:
        """
        Forward pass of the module with quantization ops applied.

        Args:
            input (Tensor): input tensor to be passed through the module
            *args (Optional[Tuple]): variable length of args
            **kwargs (Optional[Dict]): variable length of kwargs
        """
        _dtype, _device = input.dtype, input.device
        if hasattr(self, "weight") and self.weight is not None:
            _device = self.weight.device
        if self.smoothquant is not None:
            if self.smoothquant.dynamic[0] == 1 or self.smoothquant.calibrating:
                self.update_smoothquant_scale(input)
            input = self.smoothquant.scale_input(input)
        _input, args, kwags = self.input_casts(input, *args, **kwags)
        if self.obc is not None:
            self.obc.measure_hessian(_input)
        _input, args, kwags = self.align_device(_input, args, kwags, _device)
        _output = self._forward(_input, *args, **kwags)
        output = self.output_cast(_output)
        if self.flop_counter_enabled:
            self.count_flops(input, output)
        if self.align_boundary_dtype:
            output = output.to(_dtype)
        return output

    def align_device(self, _input, args, kwags, _device):
        _input = _input.to(_device) if _input.device != _device else _input
        args = tuple(
            x.to(_device) if isinstance(x, torch.Tensor) and x.device != _device else x
            for x in args
        )
        for k in kwags.keys():
            if isinstance(kwags[k], Tensor) and kwags[k].device != _device:
                kwags[k] = kwags[k].to(_device)
        return _input, args, kwags

    def update_params_with_raw(self, raw: torch.nn.Module) -> None:
        """
        Update parameters of a DmxModule from a torch.nn.Module.

        Args:
            raw (torch.nn.Module): the torch module to copy parameters from.
        """
        self.load_state_dict(raw.state_dict(), assign=True)
        # Inherit device from raw module
        for n, m in raw.named_parameters():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            device = m.device if m.device != "cpu" else device
            setattr(self, n, torch.nn.Parameter(getattr(self, n).to(device)))
        # inherit some module attributes from raw module
        self.training = raw.training
        if hasattr(raw, "dtype"):
            self.dtype = raw.dtype

    @abstractmethod
    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        pass


class DmxModuleConfig(dict):
    @classmethod
    def from_module(cls, module: DmxModule, freeze=False):
        """
        A function that stores state and ops format of the module in a DmxModuleConfig object

        Args:
            module (DmxModule): Target module for creating the DmxModuleConfig

        Returns:
            A DmxModuleConfig object that stores state and ops format of the module in a DmxModuleConfig object
        """
        cc = SimpleNamespace()
        cc.instance = module.__class__
        if isinstance(module, DmxModule):
            if module.input_formats is not None and (
                freeze
                or not all(isinstance(f, Same) for f in module.input_formats.values())
            ):
                cc.input_formats = module.input_formats
            if module.output_format is not None and (
                freeze or not isinstance(module.output_format, Same)
            ):
                cc.output_format = module.output_format
            if module.accum_format is not None and (
                freeze or not isinstance(module.accum_format, Same)
            ):
                cc.accum_format = module.accum_format
            if module.weight_format is not None and (
                freeze or not isinstance(module.weight_format, Same)
            ):
                cc.weight_format = module.weight_format
            if module.bias_format is not None and (
                freeze or not isinstance(module.bias_format, Same)
            ):
                cc.bias_format = module.bias_format
            if module.smoothquant is not None and (
                freeze or not isinstance(module.smoothquant.scale_cast.format, Same)
            ):
                cc.smoothquant_scale_format = module.smoothquant.scale_cast.format
            if module.weight_sparseness is not None and (
                freeze or not isinstance(module.weight_sparseness, Dense)
            ):
                cc.weight_sparseness = module.weight_sparseness
            if freeze or not isinstance(module.approximation_function, NoApproximation):
                cc.approximation_function = module.approximation_function
            if module.state_dict_url is not None:
                cc.state_dict_url = module.state_dict_url
        return cc.__dict__


def is_configurable(m):
    return isinstance(m, DmxModule)


class ResAdd(DmxModule, torch.nn.Module):
    """
    A module for handling residual connections.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_casts = CastToDict(
            OrderedDict({"input_cast": CastTo(), "residual_cast": CastTo()})
        )

    def forward(self, input, residual):
        if isinstance(input, torch.Tensor) and isinstance(residual, torch.Tensor):
            return DmxModule.forward(self, input, residual)
        else:
            return torch.add(input, residual)

    def _forward(self, _input: Tensor, _residual: Tensor) -> Tensor:
        """
        A forward pass of addition operation with quantization applied

        Args:
            _input (Tensor): already quantized input tensor
            residual (Tensor): residual tensor

        Returns:
            Sum of _input tensor and quantized residual tensor.
        """
        _output = _input + _residual
        return _output

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            _input = g.placeholder("_input")
            _input_scale = g.get_attr("input_casts.input_cast.scale")
            _input_zero_point = g.get_attr("input_casts.input_cast.zero_point")
            _input_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _input,
                    _input_scale,
                    _input_zero_point,
                    repr(self.input_casts.input_cast.format),
                ),
            )
            _input_dq = g.call_function(
                torch.ops.dmx.dequantize, (_input_q, _input_scale, _input_zero_point)
            )
            residual = g.placeholder("residual")
            residual_scale = g.get_attr("input_casts.residual_cast.scale")
            residual_zero_point = g.get_attr("input_casts.residual_cast.zero_point")
            residual_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    residual,
                    residual_scale,
                    residual_zero_point,
                    repr(self.input_casts.residual_cast.format),
                ),
            )
            residual_dq = g.call_function(
                torch.ops.dmx.dequantize,
                (residual_q, residual_scale, residual_zero_point),
            )
            _output = g.create_node(
                "call_function", torch.add, (_input_dq, residual_dq), name="output"
            )
            _output_scale = g.get_attr("output_cast.scale")
            _output_zero_point = g.get_attr("output_cast.zero_point")
            _output_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _output,
                    _output_scale,
                    _output_zero_point,
                    repr(self.output_cast.format),
                ),
            )
            _output_dq = g.call_function(
                torch.ops.dmx.dequantize, (_output_q, _output_scale, _output_zero_point)
            )
            g.output(_output_dq)
        return g


class Mul(DmxModule):
    def __init__(self) -> None:
        super().__init__()
        self.input_casts = CastToDict(
            OrderedDict(
                {
                    "input_cast": CastTo(block_dim=-1),
                    "multiplier_cast": CastTo(block_dim=-2),
                }
            )
        )

    def forward(self, input, multiplier):
        if isinstance(input, torch.Tensor) and isinstance(multiplier, torch.Tensor):
            return DmxModule.forward(self, input, multiplier)
        else:
            return torch.mul(input, multiplier)

    def _forward(self, _input: Tensor, multiplier: Tensor) -> Tensor:
        return _input * multiplier

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            _input = g.placeholder("_input")
            _input_scale = g.get_attr("input_casts.input_cast.scale")
            _input_zero_point = g.get_attr("input_casts.input_cast.zero_point")
            _input_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _input,
                    _input_scale,
                    _input_zero_point,
                    repr(self.input_casts.input_cast.format),
                ),
            )
            _input_dq = g.call_function(
                torch.ops.dmx.dequantize, (_input_q, _input_scale, _input_zero_point)
            )
            multiplier = g.placeholder("multiplier")
            multiplier_scale = g.get_attr("input_casts.multiplier_cast.scale")
            multiplier_zero_point = g.get_attr("input_casts.multiplier_cast.zero_point")
            multiplier_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    multiplier,
                    multiplier_scale,
                    multiplier_zero_point,
                    repr(self.input_casts.multiplier_cast.format),
                ),
            )
            multiplier_dq = g.call_function(
                torch.ops.dmx.dequantize,
                (multiplier_q, multiplier_scale, multiplier_zero_point),
            )
            _output = g.create_node(
                "call_function", torch.mul, (_input_dq, multiplier_dq), name="output"
            )
            _output_scale = g.get_attr("output_cast.scale")
            _output_zero_point = g.get_attr("output_cast.zero_point")
            _output_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _output,
                    _output_scale,
                    _output_zero_point,
                    repr(self.output_cast.format),
                ),
            )
            _output_dq = g.call_function(
                torch.ops.dmx.dequantize, (_output_q, _output_scale, _output_zero_point)
            )
            g.output(_output_dq)
        return g


class ScaledDotProductAttention(DmxModule):
    is_compound = True

    def __init__(self, dropout_p=0.0) -> None:
        super().__init__()
        self.input_casts = CastToDict(
            OrderedDict(
                {
                    "query_states_cast": CastTo(block_dim=-1),
                    "key_states_cast": CastTo(block_dim=-1),
                    "value_states_cast": CastTo(block_dim=-1),
                    "attn_mask_cast": CastTo(block_dim=-1),
                }
            )
        )
        self.add = ResAdd()
        self.matmul = ActActMatMul()
        self.softmax = Softmax(dim=-1)
        self.dropout = Dropout(p=dropout_p)

    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype)

        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = self.add(attn_bias, attn_mask)

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        attn_weight = self.matmul(query, key.transpose(-2, -1) * scale_factor)
        attn_weight = self.add(attn_weight, attn_bias)
        attn_weight = self.softmax(attn_weight)
        attn_weight = self.dropout(attn_weight)
        return self.matmul(attn_weight, value)

    def module_graph(self, *args, **kwargs) -> Graph:
        from dmx.compressor.modeling import DmxModel
        from dmx.compressor.fx.tracer import hf_symbolic_trace

        input_names, concrete_args, dummy_inputs = DmxModel.prepare_tracing_inputs(
            self, args, kwargs
        )
        gm, tracer = hf_symbolic_trace(
            self,
            input_names,
            concrete_args=concrete_args,
            dummy_inputs=dummy_inputs,
        )
        return gm

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():

            value_states = g.placeholder("value_states")
            value_states_scale = g.get_attr("input_casts.value_states_cast.scale")
            value_states_zero_point = g.get_attr(
                "input_casts.value_states_cast.zero_point"
            )
            value_states_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    value_states,
                    value_states_scale,
                    value_states_zero_point,
                    repr(self.input_casts.value_states_cast.format),
                ),
            )
            value_states_dq = g.call_function(
                torch.ops.dmx.dequantize,
                (value_states_q, value_states_scale, value_states_zero_point),
            )

            query_states = g.placeholder("query_states")
            query_states_scale = g.get_attr("input_casts.query_states_cast.scale")
            query_states_zero_point = g.get_attr(
                "input_casts.query_states_cast.zero_point"
            )
            query_states_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    query_states,
                    query_states_scale,
                    query_states_zero_point,
                    repr(self.input_casts.query_states_cast.format),
                ),
            )
            query_states_dq = g.call_function(
                torch.ops.dmx.dequantize,
                (query_states_q, query_states_scale, query_states_zero_point),
            )

            key_states = g.placeholder("key_states")
            key_states_scale = g.get_attr("input_casts.key_states_cast.scale")
            key_states_zero_point = g.get_attr("input_casts.key_states_cast.zero_point")
            key_states_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    key_states,
                    key_states_scale,
                    key_states_zero_point,
                    repr(self.input_casts.key_states_cast.format),
                ),
            )
            key_states_dq = g.call_function(
                torch.ops.dmx.dequantize,
                (key_states_q, key_states_scale, key_states_zero_point),
            )

            mask_states = g.placeholder("mask_states")
            mask_states_scale = g.get_attr("input_casts.attn_mask_cast.scale")
            mask_states_zero_point = g.get_attr("input_casts.attn_mask_cast.zero_point")
            mask_states_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    mask_states,
                    mask_states_scale,
                    mask_states_zero_point,
                    repr(self.input_casts.attn_mask_cast.format),
                ),
            )
            mask_states_dq = g.call_function(
                torch.ops.dmx.dequantize,
                (mask_states_q, mask_states_scale, mask_states_zero_point),
            )

            _output = g.create_node(
                "call_function",
                torch.nn.functional.scaled_dot_product_attention,
                (
                    value_states_dq,
                    query_states_dq,
                    key_states_dq,
                    mask_states_dq,
                ),
                name="output",
            )
            _output_scale = g.get_attr("output_cast.scale")
            _output_zero_point = g.get_attr("output_cast.zero_point")
            _output_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _output,
                    _output_scale,
                    _output_zero_point,
                    repr(self.output_cast.format),
                ),
            )
            _output_dq = g.call_function(
                torch.ops.dmx.dequantize, (_output_q, _output_scale, _output_zero_point)
            )
            g.output(_output_dq)
        return g


class ActActMatMul(DmxModule, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_casts = CastToDict(
            OrderedDict(
                {
                    "input_cast": CastTo(block_dim=-1),
                    "multiplier_cast": CastTo(block_dim=-2),
                }
            )
        )

    def forward(self, input, multiplier):
        if isinstance(input, torch.Tensor) and isinstance(multiplier, torch.Tensor):
            return DmxModule.forward(self, input, multiplier)
        else:
            return torch.matmul(input, multiplier)

    def _forward(self, _input: Tensor, _multiplier: Tensor) -> Tensor:
        _output = torch.matmul(_input, _multiplier)
        return _output

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            _input = g.placeholder("_input")
            _input_scale = g.get_attr("input_casts.input_cast.scale")
            _input_zero_point = g.get_attr("input_casts.input_cast.zero_point")
            _input_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _input,
                    _input_scale,
                    _input_zero_point,
                    repr(self.input_casts.input_cast.format),
                ),
            )
            _input_dq = g.call_function(
                torch.ops.dmx.dequantize, (_input_q, _input_scale, _input_zero_point)
            )
            multiplier = g.placeholder("multiplier")
            multiplier_scale = g.get_attr("input_casts.multiplier_cast.scale")
            multiplier_zero_point = g.get_attr("input_casts.multiplier_cast.zero_point")
            multiplier_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    multiplier,
                    multiplier_scale,
                    multiplier_zero_point,
                    repr(self.input_casts.multiplier_cast.format),
                ),
            )
            multiplier_dq = g.call_function(
                torch.ops.dmx.dequantize,
                (multiplier_q, multiplier_scale, multiplier_zero_point),
            )
            _output = g.create_node(
                "call_function", torch.matmul, (_input_dq, multiplier_dq), name="output"
            )
            _output_scale = g.get_attr("output_cast.scale")
            _output_zero_point = g.get_attr("output_cast.zero_point")
            _output_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _output,
                    _output_scale,
                    _output_zero_point,
                    repr(self.output_cast.format),
                ),
            )
            _output_dq = g.call_function(
                torch.ops.dmx.dequantize, (_output_q, _output_scale, _output_zero_point)
            )
            g.output(_output_dq)
        return g


class BAddBMM(DmxModule):
    def __init__(self) -> None:
        super().__init__()
        self.input_casts = CastToDict(
            OrderedDict(
                {
                    "input_cast": CastTo(block_dim=-1),
                    "batch1_cast": CastTo(block_dim=-1),
                    "batch2_cast": CastTo(block_dim=-2),
                }
            )
        )

    def _forward(self, input, batch1, batch2, **kwargs):
        return torch.baddbmm(input, batch1, batch2, **kwargs)


class Linear(DmxModule, torch.nn.Linear):
    r"""
    An extension of PyTorch's Linear layer to support DmxModule configurations.
    This module performs a linear transformation on the input data.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
        **kwargs: Additional keyword arguments inherited from DmxModule.

    Attributes:
        _weight (Tensor): The learnable weights of the module of shape (out_features, in_features).
        _bias (Tensor): The learnable bias of the module of shape (out_features).

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        self.input_casts.input_cast.block_dim = -1
        self.weight_cast.block_dim = -1
        if self.bias_cast is not None:
            self.bias_cast.block_dim = -1

    def _forward(self, _input: Tensor) -> Tensor:
        if isinstance(self.accum_format, Same):
            _weight = self._weight.to(_input.dtype)
            _bias = None if self._bias is None else self._bias.to(_input.dtype)
            _output = torch.nn.functional.linear(_input, _weight, _bias)
        else:
            _weight = self._weight
            _product = self.accum_cast(
                torch.matmul(_input.to(_weight.dtype), _weight.t())
            )
            if self.bias is not None:
                _output = torch.add(_product, self._bias)
            else:
                _output = _product
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        if isinstance(raw, transformers.pytorch_utils.Conv1D):
            initial_dmx = Linear(
                raw.weight.shape[0], raw.weight.shape[1], bias=raw.bias is not None
            )
            initial_dmx.weight.data = raw.weight.data.t()
            initial_dmx.bias = raw.bias
        else:
            initial_dmx = Linear(
                raw.in_features, raw.out_features, bias=raw.bias is not None
            )
            initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph

        >>> Reference:
            opcode         name                               target                             args                                                                               kwargs
            -------------  ---------------------------------  ---------------------------------  ---------------------------------------------------------------------------------  --------
            placeholder    _input                             _input                             ()                                                                                 {}
            get_attr       input_casts_input_cast_scale       input_casts.input_cast.scale       ()                                                                                 {}
            get_attr       input_casts_input_cast_zero_point  input_casts.input_cast.zero_point  ()                                                                                 {}
            call_function  quantize                           dmx.quantize                       (_input, input_casts_input_cast_scale, input_casts_input_cast_zero_point, 'SAME')  {}
            call_function  dequantize                         dmx.dequantize                     (quantize, input_casts_input_cast_scale, input_casts_input_cast_zero_point)        {}
            get_attr       _weight                            _weight                            ()                                                                                 {}
            get_attr       weight_scale                       weight_scale                       ()                                                                                 {}
            get_attr       weight_zero_point                  weight_zero_point                  ()                                                                                 {}
            call_function  quantize_1                         dmx.quantize                       (_weight, weight_scale, weight_zero_point, 'SAME')                                 {}
            call_function  dequantize_1                       dmx.dequantize                     (quantize_1, weight_scale, weight_zero_point)                                      {}
            get_attr       _bias                              _bias                              ()                                                                                 {}
            get_attr       bias_cast_scale                    bias_cast.scale                    ()                                                                                 {}
            get_attr       bias_cast_zero_point               bias_cast.zero_point               ()                                                                                 {}
            call_function  quantize_2                         dmx.quantize                       (_bias, bias_cast_scale, bias_cast_zero_point, 'SAME')                             {}
            call_function  dequantize_2                       dmx.dequantize                     (quantize_2, bias_cast_scale, bias_cast_zero_point)                                {}
            call_function  _output                            <built-in function linear>         (dequantize, dequantize_1, dequantize_2)                                           {}
            get_attr       output_cast_scale                  output_cast.scale                  ()                                                                                 {}
            get_attr       output_cast_zero_point             output_cast.zero_point             ()                                                                                 {}
            call_function  quantize_3                         dmx.quantize                       (_output, output_cast_scale, output_cast_zero_point, 'SAME')                       {}
            call_function  dequantize_3                       dmx.dequantize                     (quantize_3, output_cast_scale, output_cast_zero_point)                            {}
            output         output                             output                             (dequantize_3,)                                                                    {}

        """
        g = torch.fx.Graph()
        with g.inserting_after():
            # PLACEHOLDERS
            _input = g.placeholder("_input")
            _input_scale = g.get_attr("input_casts.input_cast.scale")
            _input_zero_point = g.get_attr("input_casts.input_cast.zero_point")
            _input_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _input,
                    _input_scale,
                    _input_zero_point,
                    repr(self.input_casts.input_cast.format),
                ),
            )
            _input_dq = g.call_function(
                torch.ops.dmx.dequantize, (_input_q, _input_scale, _input_zero_point)
            )

            # ATTRIBUTES

            # _weight
            _weight = g.get_attr("_weight")
            _weight_scale = g.get_attr("weight_scale")
            _weight_zero_point = g.get_attr("weight_zero_point")
            _weight_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _weight,
                    _weight_scale,
                    _weight_zero_point,
                    repr(self.weight_cast.format),
                ),
            )
            _weight_dq = g.call_function(
                torch.ops.dmx.dequantize, (_weight_q, _weight_scale, _weight_zero_point)
            )

            # _bias
            if self.bias is not None:
                _bias = g.get_attr("_bias")
                _bias_scale = g.get_attr("bias_cast.scale")
                _bias_zero_point = g.get_attr("bias_cast.zero_point")
                _bias_q = g.call_function(
                    torch.ops.dmx.quantize,
                    (_bias, _bias_scale, _bias_zero_point, repr(self.bias_cast.format)),
                )
                _bias_dq = g.call_function(
                    torch.ops.dmx.dequantize, (_bias_q, _bias_scale, _bias_zero_point)
                )
                _output = g.create_node(
                    "call_function",
                    torch.nn.functional.linear,
                    (_input_dq, _weight_dq, _bias_dq),
                    name="_output",
                )
                _output_scale = g.get_attr("output_cast.scale")
                _output_zero_point = g.get_attr("output_cast.zero_point")
                _output_q = g.call_function(
                    torch.ops.dmx.quantize,
                    (
                        _output,
                        _output_scale,
                        _output_zero_point,
                        repr(self.output_cast.format),
                    ),
                )
                _output_dq = g.call_function(
                    torch.ops.dmx.dequantize,
                    (_output_q, _output_scale, _output_zero_point),
                )
                g.output(_output_dq)
            else:
                _output = g.create_node(
                    "call_function",
                    torch.nn.functional.linear,
                    (_input_dq, _weight_dq, None),
                    name="_output",
                )
                _output_scale = g.get_attr("output_cast.scale")
                _output_zero_point = g.get_attr("output_cast.zero_point")
                _output_q = g.call_function(
                    torch.ops.dmx.quantize,
                    (
                        _output,
                        _output_scale,
                        _output_zero_point,
                        repr(self.output_cast.format),
                    ),
                )
                _output_dq = g.call_function(
                    torch.ops.dmx.dequantize,
                    (_output_q, _output_scale, _output_zero_point),
                )
                g.output(_output_dq)
        return g


class Embedding(DmxModule, torch.nn.Embedding):
    r"""
    An extension of PyTorch's Embedding layer to support DmxModule configurations.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        **kwargs: Additional keyword arguments inherited from torch.nn.Embedding.

    Attributes:
        _weight (Tensor):the learnable weights of the module of shape (num_embeddings, embedding_dim).

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the embedding layer.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        **kwargs,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.align_boundary_dtype = False  # special treatment for sparse layers

    def _forward(self, _input: Tensor) -> Tensor:
        _output = F.embedding(
            _input,
            self._weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        initial_dmx = Embedding(
            num_embeddings=raw.num_embeddings,
            embedding_dim=raw.embedding_dim,
            padding_idx=raw.padding_idx,
            max_norm=raw.max_norm,
            norm_type=raw.norm_type,
            scale_grad_by_freq=raw.scale_grad_by_freq,
            sparse=raw.sparse,
        )
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        initial_dmx = torch.nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )
        self.initial_dmx_graph = symbolic_trace(initial_dmx).graph
        graph = self.initial_dmx_graph
        return graph


class Conv1d(DmxModule, torch.nn.Conv1d):
    r"""
    An extension of PyTorch's Conv1d layer to support DmxModule configurations.
    This module performs a 1D convolution over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
        padding_mode (str, optional): Accepted values 'zeros' and 'circular' etc. Defaults to 'zeros'.
        **kwargs: Additional keyword arguments inherited from DmxModule.

    Attributes:
        _weight (Tensor): The learnable weights of the module of shape (out_channels, in_channels, kernel_size).
        _bias (Tensor, optional): The learnable bias of the module of shape (out_channels).

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the 1D convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            **kwargs,
        )
        self.input_casts.input_cast.block_dim = 1
        self.weight_cast.block_dim = 1
        if self.bias_cast is not None:
            self.bias_cast.block_dim = -1

    def _forward(self, _input: Tensor) -> Tensor:
        _weight = self._weight
        _convolution = self.accum_cast(
            self._conv_forward(_input.to(_weight.dtype), _weight, None)
        )
        if self.bias is not None:
            _output = torch.add(_convolution, self._bias.unsqueeze(-1).unsqueeze(-1))
        else:
            _output = _convolution
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new Conv1d object (DmxModule) from a given PyTorch Conv1d layer.

        Args:
            raw (torch.nn.Module): A PyTorch Conv1d layer to be converted.

        Returns:
            DmxModule: A Conv1d object that has the same configuration as the input PyTorch Conv1d layer.
        """
        initial_dmx = Conv1d(
            raw.in_channels,
            raw.out_channels,
            raw.kernel_size,
            stride=raw.stride,
            padding=raw.padding,
            dilation=raw.dilation,
            groups=raw.groups,
            bias=raw.bias is not None,
            padding_mode=raw.padding_mode,
        )
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class Conv2d(DmxModule, torch.nn.Conv2d):
    r"""
    An extension of PyTorch's Conv2d layer to support DmxModule configurations.
    This module performs a 2D convolution over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
        padding_mode (str, optional): Accepted values 'zeros' and 'circular' etc. Defaults to 'zeros'.
        **kwargs: Additional keyword arguments inherited from DmxModule.

    Attributes:
        _weight (Tensor): The learnable weights of the module of shape (out_channels, in_channels, kernel_height, kernel_width).
        _bias (Tensor, optional): The learnable bias of the module of shape (out_channels).

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the 2D convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            **kwargs,
        )
        self.input_casts.input_cast.block_dim = 1
        self.weight_cast.block_dim = 1
        if self.bias_cast is not None:
            self.bias_cast.block_dim = -1

    def _forward(self, _input: Tensor) -> Tensor:
        _weight = self._weight
        _convolution = self.accum_cast(
            self._conv_forward(_input.to(_weight.dtype), _weight, None)
        )
        if self.bias is not None:
            _output = torch.add(_convolution, self._bias.unsqueeze(-1).unsqueeze(-1))
        else:
            _output = _convolution
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new Conv2d object (DmxModule) from a given PyTorch Conv2d layer.

        Args:
            raw (torch.nn.Module): A PyTorch Conv2d layer to be converted.

        Returns:
            DmxModule: A Conv2d object that has the same configuration as the input PyTorch Conv2d layer.
        """
        initial_dmx = Conv2d(
            raw.in_channels,
            raw.out_channels,
            raw.kernel_size,
            stride=raw.stride,
            padding=raw.padding,
            dilation=raw.dilation,
            groups=raw.groups,
            bias=raw.bias is not None,
            padding_mode=raw.padding_mode,
        )
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class ConvTranspose2d(DmxModule, torch.nn.ConvTranspose2d):
    r"""
    An extension of PyTorch's ConvTranspose2d layer to support DmxModule configurations.
    This module performs a 2D transposed convolution over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the transposed convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the transposed convolution. Defaults to 1.
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
        padding_mode (str, optional): Accepted values 'zeros'. Defaults to 'zeros'.
        **kwargs: Additional keyword arguments inherited from DmxModule.

    Attributes:
        _weight (Tensor): The learnable weights of the module of shape (in_channels, out_channels, kernel_height, kernel_width).
        _bias (Tensor, optional): The learnable bias of the module of shape (out_channels).

    Methods:
        _forward (_input: Tensor, output_size: Optional[List[int]] = None) -> Tensor: Computes the forward pass of the 2D transposed convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        padding_mode="zeros",
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            **kwargs,
        )
        self.input_casts.input_cast.block_dim = 1
        self.weight_cast.block_dim = 1
        if self.bias_cast is not None:
            self.bias_cast.block_dim = -1

    def _forward(
        self, _input: Tensor, output_size: Optional[List[int]] = None
    ) -> Tensor:
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 2
        output_padding = self._output_padding(
            _input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,
        )  # type: ignore[arg-type]

        _weight = self._weight
        _convolution = self.accum_cast(
            F.conv_transpose2d(
                _input.to(_weight.dtype),
                _weight,
                None,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
        )
        if self.bias is not None:
            _output = torch.add(_convolution, self._bias.unsqueeze(-1).unsqueeze(-1))
        else:
            _output = _convolution
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        initial_dmx = ConvTranspose2d(
            raw.in_channels,
            raw.out_channels,
            raw.kernel_size,
            stride=raw.stride,
            padding=raw.padding,
            dilation=raw.dilation,
            groups=raw.groups,
            bias=raw.bias is not None,
            padding_mode=raw.padding_mode,
        )
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class AdaptiveAvgPool2d(DmxModule, torch.nn.AdaptiveAvgPool2d):
    r"""
    An extension of PyTorch's AdaptiveAvgPool2d layer to support DmxModule configurations.
    This module applies a 2D adaptive average pooling over an input signal composed of several input planes.

    Args:
        output_size (int or tuple): The size of the output tensor after pooling.

    Attributes:
        None specific to this subclass. Inherits attributes from parent classes.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the 2D adaptive average pooling.
    """

    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new AdaptiveAvgPool2d object (DmxModule) from a given PyTorch AdaptiveAvgPool2d layer.

        Args:
            raw (torch.nn.Module): A PyTorch AdaptiveAvgPool2d layer to be converted.

        Returns:
            DmxModule: An AdaptiveAvgPool2d object that has the same configuration as the input PyTorch AdaptiveAvgPool2d layer.
        """
        initial_dmx = AdaptiveAvgPool2d(raw.output_size)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class AvgPool2d(DmxModule, torch.nn.AvgPool2d):
    r"""
    An extension of PyTorch's AvgPool2d layer to support DmxModule configurations.
    This module applies a 2D average pooling over an input signal composed of several input planes.

    Args:
        output_size (int or tuple): The size of the output tensor after pooling.

    Attributes:
        None specific to this subclass. Inherits attributes from parent classes.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the 2D average pooling.
    """

    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new AvgPool2d object (DmxModule) from a given PyTorch AvgPool2d layer.

        Args:
            raw (torch.nn.Module): A PyTorch AvgPool2d layer to be converted.

        Returns:
            DmxModule: An AvgPool2d object that has the same configuration as the input PyTorch AvgPool2d layer.
        """
        initial_dmx = AvgPool2d(raw.output_size)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class MaxPool2d(DmxModule, torch.nn.MaxPool2d):
    r"""
    An extension of PyTorch's MaxPool2d layer to support DmxModule configurations.
    This module applies a 2D max pooling over an input signal composed of several input planes.

    Args:
        kernel_size (int or tuple): Size of the window to take a max over.
        stride (int or tuple, optional): Stride of the window. Defaults to None.
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        return_indices (bool, optional): If True, will return the max indices in a second tensor. Defaults to False.
        ceil_mode (bool, optional): If True, will use ceil instead of floor to compute the output shape. Defaults to False.

    Attributes:
        None specific to this subclass. Inherits attributes from parent classes.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the 2D max pooling.
    """

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ) -> None:
        super().__init__(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new MaxPool2d object (DmxModule) from a given PyTorch MaxPool2d layer.

        Args:
            raw (torch.nn.Module): A PyTorch MaxPool2d layer to be converted.

        Returns:
            DmxModule: A MaxPool2d object that has the same configuration as the input PyTorch MaxPool2d layer.
        """
        initial_dmx = MaxPool2d(
            raw.kernel_size,
            stride=raw.stride,
            padding=raw.padding,
            dilation=raw.dilation,
            return_indices=raw.return_indices,
            ceil_mode=raw.ceil_mode,
        )
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class Softmax(DmxModule, torch.nn.Softmax):
    r"""
    An extension of PyTorch's Softmax layer to support DmxModule configurations.
    This module applies the Softmax function to an n-dimensional input tensor, normalizing the elements
    along a specified dimension such that they sum up to 1.

    Args:
        dim (int, optional): Dimension along which Softmax will be computed. Defaults to -1.

    Attributes:
        None specific to this subclass. Inherits attributes from parent classes.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the Softmax function.
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__(dim=dim)

    def _forward(self, _input: Tensor, *args, **kwargs) -> Tensor:
        _output = self.approx_forward((_input,), dim=self.dim)
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        """
        Creates a Softmax DmxModule instance from a raw torch.nn.Module instance.

        Args:
            raw (torch.nn.Module): The raw torch.nn.Module instance.

        Returns:
            DmxModule: An initialized Softmax DmxModule instance with parameters copied from the raw instance.
        """
        initial_dmx = Softmax(dim=raw.dim)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            _input = g.placeholder("_input")
            _input_scale = g.get_attr("input_casts.input_cast.scale")
            _input_zero_point = g.get_attr("input_casts.input_cast.zero_point")
            _input_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _input,
                    _input_scale,
                    _input_zero_point,
                    repr(self.input_casts.input_cast.format),
                ),
            )
            _input_dq = g.call_function(
                torch.ops.dmx.dequantize, (_input_q, _input_scale, _input_zero_point)
            )
            dim = g.get_attr("dim")
            _output = g.create_node(
                "call_function",
                torch.nn.functional.softmax,
                (_input, dim),
                name="softmax",
            )

            _output_scale = g.get_attr("output_cast.scale")
            _output_zero_point = g.get_attr("output_cast.zero_point")
            _output_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _output,
                    _output_scale,
                    _output_zero_point,
                    repr(self.output_cast.format),
                ),
            )
            _output_dq = g.call_function(
                torch.ops.dmx.dequantize,
                (_output_q, _output_scale, _output_zero_point),
            )
            g.output(_output_dq)
        return g


class LayerNorm(DmxModule, torch.nn.LayerNorm):
    r"""
    An extension of PyTorch's LayerNorm layer to support DmxModule configurations.
    This module applies layer normalization over dimensions specified by the `normalized_shape` attribute.
    The mean and standard deviation are computed over the last `D` dimensions, where `D`is the dimensionality indicated by `normalized_shape`.
    Gamma and Beta are learnable parameters if `elementwise_affine` is set to True.

    Args:
        normalized_shape (Union[int, List[int], Size]): Specifies dimensions for layer normalization.
        eps (float, optional): A value added for numerical stability. Defaults to 1e-5.
        elementwise_affine (bool, optional): Indicates if learnable affine parameters Gamma and Beta should be used. Defaults to True.

    Attributes:
        None specific to this subclass. Inherits attributes from parent classes.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the layer normalization.
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine
        )

    def _forward(self, _input: Tensor, *args, **kwargs) -> Tensor:
        _output = self.approx_forward(
            (_input,), self.normalized_shape, self._weight, self._bias, self.eps
        )
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new LayerNorm object (DmxModule) from a given PyTorch LayerNorm layer.

        Args:
            raw (torch.nn.Module): A PyTorch LayerNorm layer to be converted.

        Returns:
            DmxModule: A LayerNorm object that has the same configuration as the input PyTorch LayerNorm layer.
        """
        initial_dmx = LayerNorm(
            raw.normalized_shape, eps=raw.eps, elementwise_affine=raw.elementwise_affine
        )
        initial_dmx.update_params_with_raw(raw)
        initial_dmx.type(raw.weight.dtype)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            _input = g.placeholder("_input")

            # Tensor Attributes

            _weight = g.get_attr("_weight")
            _weight_scale = g.get_attr("weight_scale")
            _weight_zero_point = g.get_attr("weight_zero_point")
            _weight_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _weight,
                    _weight_scale,
                    _weight_zero_point,
                    repr(self.weight_cast.format),
                ),
            )
            _weight_dq = g.call_function(
                torch.ops.dmx.dequantize, (_weight_q, _weight_scale, _weight_zero_point)
            )

            _bias = g.get_attr("_bias")
            _bias_scale = g.get_attr("bias_cast.scale")
            _bias_zero_point = g.get_attr("bias_cast.zero_point")
            _bias_q = g.call_function(
                torch.ops.dmx.quantize,
                (_bias, _bias_scale, _bias_zero_point, repr(self.bias_cast.format)),
            )
            _bias_dq = g.call_function(
                torch.ops.dmx.dequantize, (_bias_q, _bias_scale, _bias_zero_point)
            )

            # Non Tensor Attributes (no need to quantize)
            normalized_shape = g.get_attr("normalized_shape")
            eps = g.get_attr("eps")

            args = ((_input), normalized_shape, _weight_q, _bias_dq)
            output = g.create_node(
                "call_function", torch.nn.functional.layer_norm, args, name="ln"
            )
            _output_scale = g.get_attr("output_cast.scale")
            _output_zero_point = g.get_attr("output_cast.zero_point")
            _output_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    output,
                    _output_scale,
                    _output_zero_point,
                    repr(self.output_cast.format),
                ),
            )
            _output_dq = g.call_function(
                torch.ops.dmx.dequantize,
                (_output_q, _output_scale, _output_zero_point),
            )
            g.output(_output_dq)
        return g


class RMSNorm(DmxModule, torch.nn.RMSNorm):
    r"""
    An extension of RMSNorm layer to support DmxModule configurations.
    This module performs RMS-based layer normalization on the input tensor.
    The layer normalization is characterized by the `hidden_size` and an optional `eps` value for numerical stability.

    Args:
        dim (int): The size of the hidden layer (number of hidden units).
        eps (float, optional): A small constant added to the denominator for numerical stability. Defaults to 1e-6.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the RMS layer normalization.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(dim, eps=eps)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new RMSNorm object (DmxModule) from a given PyTorch RMSNorm layer.

        Args:
            raw (torch.nn.Module): A PyTorch RMSNorm layer to be converted.

        Returns:
            DmxModule: A RMSNorm object that has the same configuration as the input PyTorch RMSNorm layer.
        """
        initial_dmx = RMSNorm(
            dim=raw.weight.shape[0],
            eps=raw.variance_epsilon if hasattr(raw, "variance_epsilon") else raw.eps,
        )
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():

            _weight = g.get_attr("_weight")
            _weight_scale = g.get_attr("weight_scale")
            _weight_zero_point = g.get_attr("weight_zero_point")
            _weight_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _weight,
                    _weight_scale,
                    _weight_zero_point,
                    repr(self.weight_cast.format),
                ),
            )
            _weight_dq = g.call_function(
                torch.ops.dmx.dequantize, (_weight_q, _weight_scale, _weight_zero_point)
            )

            _input = g.placeholder("_input")
            _input_scale = g.get_attr("input_casts.input_cast.scale")
            _input_zero_point = g.get_attr("input_casts.input_cast.zero_point")
            _input_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _input,
                    _input_scale,
                    _input_zero_point,
                    repr(self.input_casts.input_cast.format),
                ),
            )
            _input_dq = g.call_function(
                torch.ops.dmx.dequantize, (_input_q, _input_scale, _input_zero_point)
            )
            # Non Tensor Attributes (no need to quantize)
            normalized_shape = g.get_attr("normalized_shape")
            eps = g.get_attr("eps")

            args = ((_input_dq), normalized_shape, _weight_dq, eps)
            output = g.create_node(
                "call_function", torch.nn.functional.rms_norm, args, name="RMSNorm"
            )
            _output_scale = g.get_attr("output_cast.scale")
            _output_zero_point = g.get_attr("output_cast.zero_point")
            _output_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    output,
                    _output_scale,
                    _output_zero_point,
                    repr(self.output_cast.format),
                ),
            )
            _output_dq = g.call_function(
                torch.ops.dmx.dequantize,
                (_output_q, _output_scale, _output_zero_point),
            )
            g.output(_output_dq)
        return g


class GemmaRMSNorm(DmxModule, transformers.models.gemma.modeling_gemma.GemmaRMSNorm):
    r"""
    An extension of RMSNorm layer to support DmxModule configurations.
    This module performs RMS-based layer normalization on the input tensor.
    The layer normalization is characterized by the `hidden_size` and an optional `eps` value for numerical stability.

    Args:
        dim (int): The size of the hidden layer (number of hidden units).
        eps (float, optional): A small constant added to the denominator for numerical stability. Defaults to 1e-6.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the RMS layer normalization.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(dim, eps=eps)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new RMSNorm object (DmxModule) from a given PyTorch RMSNorm layer.

        Args:
            raw (torch.nn.Module): A PyTorch RMSNorm layer to be converted.

        Returns:
            DmxModule: A RMSNorm object that has the same configuration as the input PyTorch RMSNorm layer.
        """
        initial_dmx = GemmaRMSNorm(
            dim=raw.weight.shape[0],
            eps=raw.variance_epsilon if hasattr(raw, "variance_epsilon") else raw.eps,
        )
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class BatchNorm2d(DmxModule, torch.nn.BatchNorm2d):
    r"""
    An extension of PyTorch's BatchNorm2d layer to support DmxModule configurations.
    This module applies batch normalization over a 4D input tensor, suitable for use with 2D convolutional layers.
    The module is parameterized by the number of features, epsilon value for numerical stability, momentum for the running mean and variance, and options to use affine transformation and track running statistics.

    Args:
        num_features (int): Number of channels in the input tensor.
        eps (float, optional): A small constant added to the denominator for numerical stability. Defaults to 1e-05.
        momentum (float, optional): The momentum value for the running mean and running variance computation. Defaults to 0.1.
        affine (bool, optional): Whether to include learnable affine parameters for this layer. Defaults to True.
        track_running_stats (bool, optional): Whether to track the running mean and variance during training. Defaults to True.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the batch normalization.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def _forward(self, _input: Tensor) -> Tensor:
        self._check_input_dim(_input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        _output = F.batch_norm(
            _input,
            (
                self.running_mean
                if not self.training or self.track_running_stats
                else None
            ),
            self.running_var if not self.training or self.track_running_stats else None,
            self._weight,
            self._bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new BatchNorm2d object (DmxModule) from a given PyTorch BatchNorm2d layer.

        Args:
            raw (torch.nn.Module): A PyTorch BatchNorm2d layer to be converted.

        Returns:
            DmxModule: A BatchNorm2d object that has the same configuration as the input PyTorch BatchNorm2d layer.
        """
        initial_dmx = BatchNorm2d(
            raw.num_features,
            eps=raw.eps,
            momentum=raw.momentum,
            affine=raw.affine,
            track_running_stats=raw.track_running_stats,
        )
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            _input = g.placeholder("_input")
            _input_scale = g.get_attr("input_casts.input_cast.scale")
            _input_zero_point = g.get_attr("input_casts.input_cast.zero_point")
            _input_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _input,
                    _input_scale,
                    _input_zero_point,
                    repr(self.input_casts.input_cast.format),
                ),
            )
            _input_dq = g.call_function(
                torch.ops.dmx.dequantize, (_input_q, _input_scale, _input_zero_point)
            )

            num_groups = g.get_attr("num_groups")
            eps = g.get_attr("eps")

            args = (_input, self.num_groups, _weight, _bias, self.eps)
            _output_scale = g.get_attr("output_cast.scale")
            _output_zero_point = g.get_attr("output_cast.zero_point")
            _output = g.create_node(
                "call_function", torch.nn.functional.group_norm, args, name="GroupNorm"
            )
            _output_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _output,
                    _output_scale,
                    _output_zero_point,
                    repr(self.output_cast.format),
                ),
            )
            _output_dq = g.call_function(
                torch.ops.dmx.dequantize, (_output_q, _output_scale, _output_zero_point)
            )
            g.output(_input)
        return g


class GroupNorm(DmxModule, torch.nn.GroupNorm):
    r"""
    An extension of PyTorch's GroupNorm layer to support DmxModule configurations.
    This module applies group normalization over an input tensor, suitable for use with various types of layers.
    The module is parameterized by the number of groups, number of channels, epsilon value for numerical stability, and an option to use affine transformation.

    Args:
        num_groups (int): Number of groups to separate the channels into.
        channels (int): Number of channels in the input tensor.
        eps (float, optional): A small constant added to the denominator for numerical stability. Defaults to 1e-5.
        affine (bool, optional): Whether to include learnable affine parameters for this layer. Defaults to True.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the group normalization.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)

    def _forward(self, input: Tensor) -> Tensor:
        _weight = self._weight
        _bias = self._bias
        _input = input.to(_weight.dtype) if _weight is not None else input

        _output = F.group_norm(_input, self.num_groups, _weight, _bias, self.eps)

        return _output

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            _input = g.placeholder("_input")
            _input_scale = g.get_attr("input_casts.input_cast.scale")
            _input_zero_point = g.get_attr("input_casts.input_cast.zero_point")
            _input_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _input,
                    _input_scale,
                    _input_zero_point,
                    repr(self.input_casts.input_cast.format),
                ),
            )
            _input_dq = g.call_function(
                torch.ops.dmx.dequantize, (_input_q, _input_scale, _input_zero_point)
            )

            num_groups = g.get_attr("num_groups")
            eps = g.get_attr("eps")

            args = (_input, self.num_groups, _weight, _bias, self.eps)
            _output_scale = g.get_attr("output_cast.scale")
            _output_zero_point = g.get_attr("output_cast.zero_point")
            _output = g.create_node(
                "call_function", torch.nn.functional.group_norm, args, name="GroupNorm"
            )
            _output_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _output,
                    _output_scale,
                    _output_zero_point,
                    repr(self.output_cast.format),
                ),
            )
            _output_dq = g.call_function(
                torch.ops.dmx.dequantize, (_output_q, _output_scale, _output_zero_point)
            )
            g.output(_input)
        return g


class Dropout(DmxModule, torch.nn.Dropout):
    r"""
    An extension of PyTorch's Dropout layer to support DmxModule configurations.
    This module applies the dropout operation over the input tensor.

    Args:
        p (float, optional): The probability of an element to be zeroed. Defaults to 0.5.
        inplace (bool, optional): If set to ``True``, will do this operation in-place. Defaults to False.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the dropout layer.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p=p, inplace=inplace)

    def _forward(self, _input: Tensor, *args, **kwargs) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new Dropout object (DmxModule) from a given PyTorch Dropout layer.

        Args:
            raw (torch.nn.Module): A PyTorch Dropout layer to be converted.

        Returns:
            DmxModule: A Dropout object that has the same configuration as the input PyTorch Dropout layer.
        """
        initial_dmx = Dropout(p=raw.p, inplace=raw.inplace)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            _input = g.placeholder("_input")
            _input_scale = g.get_attr("input_casts.input_cast.scale")
            _input_zero_point = g.get_attr("input_casts.input_cast.zero_point")
            _input_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _input,
                    _input_scale,
                    _input_zero_point,
                    repr(self.input_casts.input_cast.format),
                ),
            )
            _input_dq = g.call_function(
                torch.ops.dmx.dequantize, (_input_q, _input_scale, _input_zero_point)
            )

            p = g.get_attr("p")
            training = g.get_attr("training")
            inplace = g.get_attr("inplace")

            args = (_input_dq, p, training, inplace)
            _output_scale = g.get_attr("output_cast.scale")
            _output_zero_point = g.get_attr("output_cast.zero_point")
            _output = g.create_node(
                "call_function", torch.nn.functional.dropout, args, name="Dropout"
            )
            _output_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _output,
                    _output_scale,
                    _output_zero_point,
                    repr(self.output_cast.format),
                ),
            )
            _output_dq = g.call_function(
                torch.ops.dmx.dequantize, (_output_q, _output_scale, _output_zero_point)
            )
            g.output(_input)
        return g


class ReLU(DmxModule, torch.nn.ReLU):
    r"""
    An extension of PyTorch's ReLU layer to support DmxModule configurations.
    This module applies the Rectified Linear Unit (ReLU) function element-wise on the input tensor.

    Args:
        inplace (bool, optional): If set to ``True``, will do this operation in-place. Defaults to False.

    Methods:
        _forward (_input: Tensor, inplace: bool = False) -> Tensor: Computes the forward pass of the ReLU layer.
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)

    def _forward(self, _input: Tensor, inplace: bool = False) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new ReLU object (DmxModule) from a given PyTorch ReLU layer.

        Args:
            raw (torch.nn.Module): A PyTorch ReLU layer to be converted.

        Returns:
            DmxModule: A ReLU object that has the same configuration as the input PyTorch ReLU layer.
        """
        initial_dmx = ReLU(inplace=raw.inplace)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            _input = g.placeholder("_input")
            _input_scale = g.get_attr("input_casts.input_cast.scale")
            _input_zero_point = g.get_attr("input_casts.input_cast.zero_point")
            _input_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _input,
                    _input_scale,
                    _input_zero_point,
                    repr(self.input_casts.input_cast.format),
                ),
            )
            _input_dq = g.call_function(
                torch.ops.dmx.dequantize, (_input_q, _input_scale, _input_zero_point)
            )

            args = (_input_dq,)
            _output_scale = g.get_attr("output_cast.scale")
            _output_zero_point = g.get_attr("output_cast.zero_point")
            _output = g.create_node(
                "call_function", torch.nn.functional.relu, args, name="ReLU"
            )
            _output_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _output,
                    _output_scale,
                    _output_zero_point,
                    repr(self.output_cast.format),
                ),
            )
            _output_dq = g.call_function(
                torch.ops.dmx.dequantize, (_output_q, _output_scale, _output_zero_point)
            )
            g.output(_output_dq)
        return g


class ReLU6(DmxModule, torch.nn.ReLU6):
    r"""
    An extension of PyTorch's ReLU6 layer to support DmxModule configurations.
    This module applies the Rectified Linear Unit 6 (ReLU6) function element-wise on the input tensor.

    Args:
        inplace (bool, optional): If set to ``True``, will do this operation in-place. Defaults to False.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the ReLU6 layer.
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new ReLU6 object (DmxModule) from a given PyTorch ReLU6 layer.

        Args:
            raw (torch.nn.Module): A PyTorch ReLU6 layer to be converted.
        Returns:
            DmxModule: A ReLU6 object that has the same configuration as the input PyTorch ReLU6 layer.
        """
        initial_dmx = ReLU6(inplace=raw.inplace)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            _input = g.placeholder("_input")
            _input_scale = g.get_attr("input_casts.input_cast.scale")
            _input_zero_point = g.get_attr("input_casts.input_cast.zero_point")
            _input_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _input,
                    _input_scale,
                    _input_zero_point,
                    repr(self.input_casts.input_cast.format),
                ),
            )
            _input_dq = g.call_function(
                torch.ops.dmx.dequantize, (_input_q, _input_scale, _input_zero_point)
            )

            args = _input_dq
            _output_scale = g.get_attr("output_cast.scale")
            _output_zero_point = g.get_attr("output_cast.zero_point")
            _output = g.create_node(
                "call_function", torch.nn.functional.relu6, args, name="relu6"
            )
            _output_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _output,
                    _output_scale,
                    _output_zero_point,
                    repr(self.output_cast.format),
                ),
            )
            _output_dq = g.call_function(
                torch.ops.dmx.dequantize, (_output_q, _output_scale, _output_zero_point)
            )
            g.output(_output_dq)
        return g


class SiLU(DmxModule, torch.nn.SiLU):
    r"""
    An extension of PyTorch's SiLU (Sigmoid Linear Unit) layer to support DmxModule configurations.
    This module applies the SiLU function element-wise on the input tensor.

    Args:
        inplace (bool, optional): If set to ``True``, will do this operation in-place. Defaults to False.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the SiLU layer.
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new SiLU object (DmxModule) from a given PyTorch SiLU layer.

        Args:
            raw (torch.nn.Module): A PyTorch SiLU layer to be converted.

        Returns:
            DmxModule: A SiLU object that has the same configuration as the input PyTorch SiLU layer.
        """
        initial_dmx = SiLU(inplace=raw.inplace)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            _input = g.placeholder("_input")
            _input_scale = g.get_attr("input_casts.input_cast.scale")
            _input_zero_point = g.get_attr("input_casts.input_cast.zero_point")
            _input_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _input,
                    _input_scale,
                    _input_zero_point,
                    repr(self.input_casts.input_cast.format),
                ),
            )
            _input_dq = g.call_function(
                torch.ops.dmx.dequantize, (_input_q, _input_scale, _input_zero_point)
            )

            inplace = g.get_attr("inplace")
            args = (_input_dq, inplace)
            _output_scale = g.get_attr("output_cast.scale")
            _output_zero_point = g.get_attr("output_cast.zero_point")
            _output = g.create_node(
                "call_function", torch.nn.functional.silu, args, name="SiLU"
            )
            _output_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _output,
                    _output_scale,
                    _output_zero_point,
                    repr(self.output_cast.format),
                ),
            )
            _output_dq = g.call_function(
                torch.ops.dmx.dequantize, (_output_q, _output_scale, _output_zero_point)
            )
            g.output(_output_dq)
        return g


class Tanh(DmxModule, torch.nn.Tanh):
    r"""
    An extension of PyTorch's Tanh (Hyperbolic Tangent) layer to support DmxModule configurations.
    This module applies the tanh function element-wise on the input tensor.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the Tanh layer.
    """

    def __init__(self) -> None:
        super().__init__()

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new Tanh object (DmxModule) from a given PyTorch Tanh layer.

        Args:
            raw (torch.nn.Module): A PyTorch Tanh layer to be converted.

        Returns:
            DmxModule: A Tanh object that has the same configuration as the input PyTorch Tanh layer.
        """
        initial_dmx = Tanh()
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            _input = g.placeholder("_input")
            _input_scale = g.get_attr("input_casts.input_cast.scale")
            _input_zero_point = g.get_attr("input_casts.input_cast.zero_point")
            _input_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _input,
                    _input_scale,
                    _input_zero_point,
                    repr(self.input_casts.input_cast.format),
                ),
            )
            _input_dq = g.call_function(
                torch.ops.dmx.dequantize, (_input_q, _input_scale, _input_zero_point)
            )

            args = _input_dq
            _output_scale = g.get_attr("output_cast.scale")
            _output_zero_point = g.get_attr("output_cast.zero_point")
            _output = g.create_node(
                "call_function", torch.nn.functional.tanh, args, name="tanh"
            )
            _output_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _output,
                    _output_scale,
                    _output_zero_point,
                    repr(self.output_cast.format),
                ),
            )
            _output_dq = g.call_function(
                torch.ops.dmx.dequantize, (_output_q, _output_scale, _output_zero_point)
            )
            g.output(_output_dq)
        return g


class GELUBase(DmxModule):
    r"""
    A generalized base class to support various GELUActivation configurations.
    This module applies the specified GELUActivation function element-wise on the input tensor.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the GELU layer.
    """

    def __init__(self, activation_cls, *args, **kwargs) -> None:
        if activation_cls not in self.__class__.__bases__:
            self.__class__.__bases__ += (activation_cls,)
        super().__init__(*args, **kwargs)
        self.activation_cls = activation_cls

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new GELU object (DmxModule) from a given Transformers layer.

        Args:
            raw (torch.nn.Module): A Transformers GELUActivation layer to be converted.

        Returns:
            DmxModule: A GELU object that has the same configuration as the input Transformers GELUActivation layer.
        """
        initial_dmx = cls()
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        initial_dmx = self.activation_cls()
        self.initial_dmx_graph = symbolic_trace(initial_dmx).graph
        graph = self.initial_dmx_graph
        return graph


class GELU(GELUBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(torch.nn.GELU, *args, **kwargs)


class NewGELU(GELUBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(transformers.activations.NewGELUActivation, *args, **kwargs)


class FastGELU(GELUBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(transformers.activations.FastGELUActivation, *args, **kwargs)


class QuickGELU(GELUBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(transformers.activations.QuickGELUActivation, *args, **kwargs)


class ClippedGELU(GELUBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            transformers.activations.ClippedGELUActivation, *args, **kwargs
        )


class BloomGELU(GELUBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            transformers.models.bloom.modeling_bloom.BloomGelu, *args, **kwargs
        )
