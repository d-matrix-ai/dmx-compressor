import torch
from torch.fx import Graph
from . import DmxModule


class Conv1d(DmxModule, torch.nn.Conv1d):
    r"""
    This is an alternative version of the DmxModule .nn.Conv1d,
    without calling torch.nn.functional.conv1d(), but torch.nn.functional.unfold() and torch.matmul() instead.
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

    def _forward(self, _input: torch.Tensor) -> torch.Tensor:
        _weight = self._weight.reshape((self.out_channels, -1))
        _input = torch.nn.functional.unfold(
            _input.unsqueeze(-1),
            kernel_size=self.kernel_size + (1,),
            dilation=self.dilation + (1,),
            padding=self.padding + (0,),
            stride=self.stride + (1,),
        )
        _convolution = self.accum_cast(torch.matmul(_weight, _input))
        if self.bias is not None:
            _output = torch.add(_convolution, self._bias.unsqueeze(-1))
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

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
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

            _unsqueeze = g.call_function(torch.unsqueeze, (_input_dq, -1))
            _unfold = g.create_node(
                "call_function",
                torch.nn.functional.unfold,
                (_unsqueeze, self.kernel_size + (1,)),
                dict(
                    dilation=self.dilation + (1,),
                    padding=self.padding + (0,),
                    stride=self.stride + (1,),
                ),
                name="_unfold",
            )

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

            _reshape = g.call_function(
                torch.reshape, (_weight_dq, (self.out_channels, -1))
            )
            _matmul = g.call_function(torch.matmul, (_reshape, _unfold))

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
            _bias_unsqueeze = g.call_function(torch.unsqueeze, (_bias_dq, -1))

            _output = g.call_function(
                torch.add,
                (_matmul, _bias_unsqueeze),
            )
            _output_scale = g.get_attr("output_casts.output_cast.scale")
            _output_zero_point = g.get_attr("output_casts.output_cast.zero_point")
            _output_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    _output,
                    _output_scale,
                    _output_zero_point,
                    repr(self.output_casts.output_cast.format),
                ),
            )
            _output_dq = g.call_function(
                torch.ops.dmx.dequantize,
                (_output_q, _output_scale, _output_zero_point),
            )
            g.output(_output_dq)
        return g

class Conv2d(DmxModule, torch.nn.Conv2d):
    r"""
    This is an alternative version of the DmxModule.nn.Conv2d,
    without calling torch.nn.functional.conv2d(), but torch.nn.functional.unfold() and torch.matmul() instead.
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

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        _, _, in_height, in_width = _input.shape
        _h_out = (in_height + 2 * self.padding[0] - (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        _w_out = (in_width + 2 * self.padding[1] - (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        _weight = self._weight.reshape((self.out_channels, -1))
        _input = torch.nn.functional.unfold(
            _input,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        _convolution = self.accum_cast(torch.matmul(_weight, _input))
        if self.bias is not None:
            _output = torch.add(_convolution, self._bias.unsqueeze(-1))
        else:
            _output = _convolution
        return torch.nn.functional.fold(_output, (_h_out, _w_out), (1, 1))

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

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        raise NotImplementedError("to_compiler_graph not implemented!")