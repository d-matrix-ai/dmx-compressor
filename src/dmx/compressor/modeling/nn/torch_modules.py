import math
from typing import Union, List, Optional
from collections import OrderedDict
import torch
from torch import Tensor, Size
import torch.nn.functional as F
from torch.fx import Graph, symbolic_trace
import transformers
import transformers.activations

from dmx.compressor.numerical import Same, CastTo, CastToDict
from . import DmxModule


class ResAdd(DmxModule, torch.nn.Module):
    """
    A module for handling residual connections.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_casts = CastToDict(
            OrderedDict({"input_cast": CastTo(), "residual_cast": CastTo()})
        )

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
            placeholder_nodes = self.create_placeholders(g, ["_input", "residual"])
            input_dq, residual_dq = self.qdq_nodes(
                g,
                placeholder_nodes,
                ["input_casts.input_cast", "input_casts.residual_cast"],
            )

            _output = g.create_node(
                "call_function", torch.add, (input_dq, residual_dq), name="output"
            )
            _output_dq = self.qdq_nodes(g, [_output], ["output_casts.output_cast"])
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

    def _forward(self, _input: Tensor, multiplier: Tensor) -> Tensor:
        return _input * multiplier

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            placeholder_nodes = self.create_placeholders(g, ["_input", "multiplier"])
            _input_dq, multiplier_dq = self.qdq_nodes(
                g,
                placeholder_nodes,
                ["input_casts.input_cast", "input_casts.multiplier_cast"],
            )
            _output = g.create_node(
                "call_function", torch.mul, (_input_dq, multiplier_dq), name="output"
            )
            _output_dq = self.qdq_nodes(g, [_output], ["output_casts.output_cast"])
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
        self.resadd = ResAdd()
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
        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)

        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = self.resadd(attn_bias, attn_mask)

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        attn_weight = self.matmul(query, key.transpose(-2, -1) * scale_factor)
        attn_weight = self.resadd(attn_weight, attn_bias)
        attn_weight = self.softmax(attn_weight)
        attn_weight = self.dropout(attn_weight)
        return self.matmul(attn_weight, value)

    def module_graph(self, *args, **kwargs) -> Graph:
        from dmx.compressor.fx.transform import prepare_tracing_inputs
        from dmx.compressor.fx.tracer import hf_symbolic_trace

        input_names, concrete_args, dummy_inputs = prepare_tracing_inputs(
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
        import warnings

        warnings.warn("SDPA is not decomposed, torch sdpa function is used.")
        g = torch.fx.Graph()
        with g.inserting_after():
            placeholders = self.create_placeholders(
                g, ["value", "query", "key", "mask"]
            )
            cast_names = [
                "input_casts.value_states_cast",
                "input_casts.query_states_cast",
                "input_casts.key_states_cast",
                "input_casts.attn_mask_cast",
            ]
            value_states_dq, query_states_dq, key_states_dq, mask_states_dq = (
                self.qdq_nodes(g, placeholders, cast_names)
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
            _output_dq = self.qdq_nodes(g, [_output], ["output_casts.output_cast"])
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

    def _forward(self, _input: Tensor, _multiplier: Tensor) -> Tensor:
        _output = torch.matmul(_input, _multiplier)
        return _output

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            placeholder_nodes = self.create_placeholders(g, ["_input", "multiplier"])
            _input_dq, multiplier_dq = self.qdq_nodes(
                g,
                placeholder_nodes,
                ["input_casts.input_cast", "input_casts.multiplier_cast"],
            )
            _output = g.create_node(
                "call_function", torch.matmul, (_input_dq, multiplier_dq), name="output"
            )
            _output_dq = self.qdq_nodes(g, [_output], ["output_casts.output_cast"])
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

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        if isinstance(raw, transformers.pytorch_utils.Conv1D):
            initial_dmx = cls(
                raw.weight.shape[0], raw.weight.shape[1], bias=raw.bias is not None
            )
            initial_dmx.weight.data = raw.weight.data.t()
            initial_dmx.bias = raw.bias
        else:
            initial_dmx = cls(
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
            get_attr       output_casts_output_cast_scale     output_casts.output_cast.scale     ()                                                                                 {}
            get_attr       output_casts_output_cast_zero_pointoutput_casts.utput_cast.zero_point ()                                                                                 {}
            call_function  quantize_3                         dmx.quantize                       (_output, output_casts_output_cast_scale, output_casts_output_cast_zero_pointoutput_casts, 'SAME')  {}
            call_function  dequantize_3                       dmx.dequantize                     (quantize_3, output_casts_output_cast_scale, output_casts_output_cast_zero_pointoutput_casts)       {}
            output         output                             output                             (dequantize_3,)                                                                    {}

        """
        g = torch.fx.Graph()
        with g.inserting_after():
            # PLACEHOLDERS
            placeholder_nodes = self.create_placeholders(g, ["_input"])
            _input_dq = self.qdq_nodes(
                g,
                placeholder_nodes,
                ["input_casts.input_cast"],
            )

            # ATTRIBUTES

            # _weight
            _weight = g.get_attr("_weight")
            _weight_dq = self.qdq_nodes(g, [_weight], ["weight_cast"])

            # _bias
            if self.bias is not None:
                _bias = g.get_attr("_bias")
                _bias_dq = self.qdq_nodes(g, [_bias], ["bias_cast"])
                _output = g.create_node(
                    "call_function",
                    torch.nn.functional.linear,
                    (_input_dq, _weight_dq, _bias_dq),
                    name="_output",
                )
                _output_dq = self.qdq_nodes(g, [_output], ["output_casts.output_cast"])
                g.output(_output_dq)
            else:
                _output = g.create_node(
                    "call_function",
                    torch.nn.functional.linear,
                    (_input_dq, _weight_dq, None),
                    name="_output",
                )
                _output_dq = self.qdq_nodes(g, [_output], ["output_casts.output_cast"])
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

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        initial_dmx = cls(
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
            _output = torch.add(_convolution, self._bias.unsqueeze(-1))
        else:
            _output = _convolution
        return _output

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new Conv1d object (DmxModule) from a given PyTorch Conv1d layer.

        Args:
            raw (torch.nn.Module): A PyTorch Conv1d layer to be converted.

        Returns:
            DmxModule: A Conv1d object that has the same configuration as the input PyTorch Conv1d layer.
        """
        initial_dmx = cls(
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

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new Conv2d object (DmxModule) from a given PyTorch Conv2d layer.

        Args:
            raw (torch.nn.Module): A PyTorch Conv2d layer to be converted.

        Returns:
            DmxModule: A Conv2d object that has the same configuration as the input PyTorch Conv2d layer.
        """
        initial_dmx = cls(
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

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        initial_dmx = cls(
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

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new AdaptiveAvgPool2d object (DmxModule) from a given PyTorch AdaptiveAvgPool2d layer.

        Args:
            raw (torch.nn.Module): A PyTorch AdaptiveAvgPool2d layer to be converted.

        Returns:
            DmxModule: An AdaptiveAvgPool2d object that has the same configuration as the input PyTorch AdaptiveAvgPool2d layer.
        """
        initial_dmx = cls(raw.output_size)
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

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new AvgPool2d object (DmxModule) from a given PyTorch AvgPool2d layer.

        Args:
            raw (torch.nn.Module): A PyTorch AvgPool2d layer to be converted.

        Returns:
            DmxModule: An AvgPool2d object that has the same configuration as the input PyTorch AvgPool2d layer.
        """
        initial_dmx = cls(raw.output_size)
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

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new MaxPool2d object (DmxModule) from a given PyTorch MaxPool2d layer.

        Args:
            raw (torch.nn.Module): A PyTorch MaxPool2d layer to be converted.

        Returns:
            DmxModule: A MaxPool2d object that has the same configuration as the input PyTorch MaxPool2d layer.
        """
        initial_dmx = cls(
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

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        """
        Creates a Softmax DmxModule instance from a raw torch.nn.Module instance.

        Args:
            raw (torch.nn.Module): The raw torch.nn.Module instance.

        Returns:
            DmxModule: An initialized Softmax DmxModule instance with parameters copied from the raw instance.
        """
        initial_dmx = cls(dim=raw.dim)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            placeholder_nodes = self.create_placeholders(g, ["_input"])
            _input_dq = self.qdq_nodes(
                g,
                placeholder_nodes,
                ["input_casts.input_cast"],
            )
            dim = g.get_attr("dim")
            _output = g.create_node(
                "call_function",
                torch.nn.functional.softmax,
                (_input_dq, dim),
                name="softmax",
            )
            _output_dq = self.qdq_nodes(g, [_output], ["output_casts.output_cast"])
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
        self.functional_forward = F.layer_norm

    def _forward(self, _input: Tensor, *args, **kwargs) -> Tensor:
        _output = self.approx_forward(
            (_input,), self.normalized_shape, self._weight, self._bias, self.eps
        )
        return _output

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new LayerNorm object (DmxModule) from a given PyTorch LayerNorm layer.

        Args:
            raw (torch.nn.Module): A PyTorch LayerNorm layer to be converted.

        Returns:
            DmxModule: A LayerNorm object that has the same configuration as the input PyTorch LayerNorm layer.
        """
        initial_dmx = cls(
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
            placeholder_nodes = self.create_placeholders(g, ["_input"])
            _input_dq = self.qdq_nodes(
                g,
                placeholder_nodes,
                ["input_casts.input_cast"],
            )

            # Tensor Attributes

            _weight = g.get_attr("_weight")
            _weight_dq = self.qdq_nodes(g, [_weight], ["weight_cast"])

            _bias = g.get_attr("_bias")
            _bias_dq = self.qdq_nodes(g, [_bias], ["bias_cast"])

            # Non Tensor Attributes (no need to quantize)
            normalized_shape = g.get_attr("normalized_shape")
            eps = g.get_attr("eps")

            args = ((_input_dq), normalized_shape, _weight_dq, _bias_dq)
            output = g.create_node(
                "call_function", torch.nn.functional.layer_norm, args, name="ln"
            )
            _output_dq = self.qdq_nodes(g, [output], ["output_casts.output_cast"])
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
        normalized_shape: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(normalized_shape, eps=eps)
        self.functional_forward = F.rms_norm

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(
            (_input,), self.normalized_shape, self._weight, self.eps
        )
        return _output

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new RMSNorm object (DmxModule) from a given PyTorch RMSNorm layer.

        Args:
            raw (torch.nn.Module): A PyTorch RMSNorm layer to be converted.

        Returns:
            DmxModule: A RMSNorm object that has the same configuration as the input PyTorch RMSNorm layer.
        """
        initial_dmx = cls(
            normalized_shape=raw.weight.shape[0],
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
            placeholder_nodes = self.create_placeholders(g, ["_input"])
            _input_dq = self.qdq_nodes(
                g,
                placeholder_nodes,
                ["input_casts.input_cast"],
            )
            _weight = g.get_attr("_weight")
            _weight_dq = self.qdq_nodes(g, [_weight], ["weight_cast"])

            # Non Tensor Attributes (no need to quantize)
            normalized_shape = g.get_attr("normalized_shape")
            eps = g.get_attr("eps")

            args = ((_input_dq), normalized_shape, _weight_dq, eps)
            output = g.create_node(
                "call_function", torch.nn.functional.rms_norm, args, name="RMSNorm"
            )
            _output_dq = self.qdq_nodes(g, [output], ["output_casts.output_cast"])
            g.output(_output_dq)
        return g


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

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new BatchNorm2d object (DmxModule) from a given PyTorch BatchNorm2d layer.

        Args:
            raw (torch.nn.Module): A PyTorch BatchNorm2d layer to be converted.

        Returns:
            DmxModule: A BatchNorm2d object that has the same configuration as the input PyTorch BatchNorm2d layer.
        """
        initial_dmx = cls(
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
            placeholder_nodes = self.create_placeholders(g, ["_input"])
            _input_dq = self.qdq_nodes(
                g,
                placeholder_nodes,
                ["input_casts.input_cast"],
            )
            _weight = g.get_attr("_weight")
            _weight_dq = self.qdq_nodes(g, [_weight], ["weight_cast"])
            _bias = g.get_attr("_bias")
            _bias_dq = self.qdq_nodes(g, [_bias], ["bias_cast"])

            args = (_input_dq, self.num_groups, _weight_dq, _bias_dq, self.eps)
            _output = g.create_node(
                "call_function", torch.nn.functional.group_norm, args, name="GroupNorm"
            )
            _output_dq = self.qdq_nodes(g, [_output], ["output_casts.output_cast"])
            g.output(_output_dq)
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
            placeholder_nodes = self.create_placeholders(g, ["_input"])
            _input_dq = self.qdq_nodes(
                g,
                placeholder_nodes,
                ["input_casts.input_cast"],
            )

            _weight = g.get_attr("_weight")
            _weight_dq = self.qdq_nodes(g, [_weight], ["weight_cast"])
            _bias = g.get_attr("_bias")
            _bias_dq = self.qdq_nodes(g, [_bias], ["bias_cast"])

            args = (_input_dq, self.num_groups, _weight_dq, _bias_dq, self.eps)

            _output = g.create_node(
                "call_function", torch.nn.functional.group_norm, args, name="GroupNorm"
            )
            _output_dq = self.qdq_nodes(g, [_output], ["output_casts.output_cast"])
            g.output(_output_dq)
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

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new Dropout object (DmxModule) from a given PyTorch Dropout layer.

        Args:
            raw (torch.nn.Module): A PyTorch Dropout layer to be converted.

        Returns:
            DmxModule: A Dropout object that has the same configuration as the input PyTorch Dropout layer.
        """
        initial_dmx = cls(p=raw.p, inplace=raw.inplace)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            placeholder_nodes = self.create_placeholders(g, ["_input"])
            _input_dq = self.qdq_nodes(
                g,
                placeholder_nodes,
                ["input_casts.input_cast"],
            )

            p = g.get_attr("p")
            training = g.get_attr("training")
            inplace = g.get_attr("inplace")

            args = (_input_dq, p, training, inplace)
            _output = g.create_node(
                "call_function", torch.nn.functional.dropout, args, name="Dropout"
            )
            _output_dq = self.qdq_nodes(g, [_output], ["output_casts.output_cast"])
            g.output(_output_dq)
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

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new ReLU object (DmxModule) from a given PyTorch ReLU layer.

        Args:
            raw (torch.nn.Module): A PyTorch ReLU layer to be converted.

        Returns:
            DmxModule: A ReLU object that has the same configuration as the input PyTorch ReLU layer.
        """
        initial_dmx = cls(inplace=raw.inplace)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            placeholder_nodes = self.create_placeholders(g, ["_input"])
            _input_dq = self.qdq_nodes(
                g,
                placeholder_nodes,
                ["input_casts.input_cast"],
            )

            args = (_input_dq,)
            _output = g.create_node(
                "call_function", torch.nn.functional.relu, args, name="ReLU"
            )
            _output_dq = self.qdq_nodes(g, [_output], ["output_casts.output_cast"])
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

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new ReLU6 object (DmxModule) from a given PyTorch ReLU6 layer.

        Args:
            raw (torch.nn.Module): A PyTorch ReLU6 layer to be converted.
        Returns:
            DmxModule: A ReLU6 object that has the same configuration as the input PyTorch ReLU6 layer.
        """
        initial_dmx = cls(inplace=raw.inplace)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            placeholder_nodes = self.create_placeholders(g, ["_input"])
            _input_dq = self.qdq_nodes(
                g,
                placeholder_nodes,
                ["input_casts.input_cast"],
            )

            args = _input_dq
            _output = g.create_node(
                "call_function", torch.nn.functional.relu6, args, name="relu6"
            )
            _output_dq = self.qdq_nodes(g, [_output], ["output_casts.output_cast"])
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

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new SiLU object (DmxModule) from a given PyTorch SiLU layer.

        Args:
            raw (torch.nn.Module): A PyTorch SiLU layer to be converted.

        Returns:
            DmxModule: A SiLU object that has the same configuration as the input PyTorch SiLU layer.
        """
        initial_dmx = cls(inplace=raw.inplace)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            placeholder_nodes = self.create_placeholders(g, ["_input"])
            _input_dq = self.qdq_nodes(
                g,
                placeholder_nodes,
                ["input_casts.input_cast"],
            )

            inplace = g.get_attr("inplace")
            args = (_input_dq, inplace)
            _output = g.create_node(
                "call_function", torch.nn.functional.silu, args, name="SiLU"
            )
            _output_dq = self.qdq_nodes(g, [_output], ["output_casts.output_cast"])
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

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new Tanh object (DmxModule) from a given PyTorch Tanh layer.

        Args:
            raw (torch.nn.Module): A PyTorch Tanh layer to be converted.

        Returns:
            DmxModule: A Tanh object that has the same configuration as the input PyTorch Tanh layer.
        """
        initial_dmx = cls()
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = torch.fx.Graph()
        with g.inserting_after():
            placeholder_nodes = self.create_placeholders(g, ["_input"])
            _input_dq = self.qdq_nodes(
                g,
                placeholder_nodes,
                ["input_casts.input_cast"],
            )
            args = _input_dq

            _output = g.create_node(
                "call_function", torch.nn.functional.tanh, args, name="tanh"
            )
            _output_dq = self.qdq_nodes(g, [_output], ["output_casts.output_cast"])
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
