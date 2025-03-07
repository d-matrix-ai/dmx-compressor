import torch
from torch.fx import Graph
from . import DmxModule, Conv1d as _Conv1d, Conv2d as _Conv2d


class Conv1d(_Conv1d):
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
            matmul = g.call_function(torch.matmul, (_reshape, _unfold))
            _matmul_scale = g.get_attr("accum_cast.scale")
            _matmul_zero_point = g.get_attr("accum_cast.zero_point")
            _matmul_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    matmul,
                    _matmul_scale,
                    _matmul_zero_point,
                    repr(self.accum_cast.format),
                ),
            )
            _matmul_dq = g.call_function(
                torch.ops.dmx.dequantize, (_matmul_q, _matmul_scale, _matmul_zero_point)
            )

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
                _bias_unsqueeze = g.call_function(torch.unsqueeze, (_bias_dq, -1))

                _output = g.call_function(
                    torch.add,
                    (_matmul_dq, _bias_unsqueeze),
                )
            else:
                _output = _matmul_dq

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


class Conv2d(_Conv2d):
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

    def _forward(self, _input: torch.Tensor) -> torch.Tensor:
        _, _, in_height, in_width = _input.shape
        _h_out = (
            in_height + 2 * self.padding[0] - (self.kernel_size[0] - 1) - 1
        ) // self.stride[0] + 1
        _w_out = (
            in_width + 2 * self.padding[1] - (self.kernel_size[1] - 1) - 1
        ) // self.stride[1] + 1
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

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        import operator

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
            getattr_1 = g.call_function(getattr, (_input_dq, "shape"), {})
            getitem_2 = g.call_function(operator.getitem, (getattr_1, 2), {})
            getitem_3 = g.call_function(operator.getitem, (getattr_1, 3), {})
            add = g.call_function(operator.add, (getitem_2, 2 * self.padding[0]), {})
            sub = g.call_function(operator.sub, (add, self.kernel_size[0] - 1), {})
            sub_1 = g.call_function(operator.sub, (sub, 1), {})
            floordiv = g.call_function(operator.floordiv, (sub_1, self.stride[0]), {})
            add_1 = g.call_function(operator.add, (floordiv, 1), {})
            add_2 = g.call_function(operator.add, (getitem_3, 2 * self.padding[1]), {})
            sub_2 = g.call_function(operator.sub, (add_2, self.kernel_size[1] - 1), {})
            sub_3 = g.call_function(operator.sub, (sub_2, 1), {})
            floordiv_1 = g.call_function(operator.floordiv, (sub_3, self.stride[1]), {})
            add_3 = g.call_function(operator.add, (floordiv_1, 1), {})

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

            getattr_2 = g.call_function(getattr, (_weight, "device"), {})
            to = g.call_method("to", (_weight_dq, getattr_2), {})
            clone = g.call_method("clone", (to,), {})
            getattr_3 = g.call_function(getattr, (to, "dtype"), {})
            to_1 = g.call_method("to", (clone, getattr_3), {})
            reshape = g.call_method("reshape", (to_1, (self.out_channels, -1)), {})
            unfold = g.call_function(
                torch.nn.functional.unfold,
                (_input_dq, self.kernel_size),
                {
                    "dilation": self.dilation,
                    "padding": self.padding,
                    "stride": self.stride,
                },
            )
            matmul = g.call_function(torch.matmul, (reshape, unfold), {})
            _matmul_scale = g.get_attr("accum_cast.scale")
            _matmul_zero_point = g.get_attr("accum_cast.zero_point")
            _matmul_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    matmul,
                    _matmul_scale,
                    _matmul_zero_point,
                    repr(self.accum_cast.format),
                ),
            )
            _matmul_dq = g.call_function(
                torch.ops.dmx.dequantize, (_matmul_q, _matmul_scale, _matmul_zero_point)
            )
            clone_1 = g.call_method("clone", (_matmul_dq,), {})
            getattr_4 = g.call_function(getattr, (matmul, "dtype"), {})
            to_2 = g.call_method("to", (clone_1, getattr_4), {})
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
                clone_2 = g.call_method("clone", (_bias_dq,), {})
                getattr_5 = g.call_function(getattr, (_bias, "dtype"), {})
                to_3 = g.call_method("to", (clone_2, getattr_5), {})
                unsqueeze = g.call_method("unsqueeze", (to_3, -1), {})
                add_4 = g.call_function(torch.add, (to_2, unsqueeze), {})
                output = add_4
            else:
                output = to_2
            fold = g.call_function(
                torch.nn.functional.fold,
                (output, (add_1, add_3), (1, 1)),
            )
            _output_scale = g.get_attr("output_casts.output_cast.scale")
            _output_zero_point = g.get_attr("output_casts.output_cast.zero_point")
            _output_q = g.call_function(
                torch.ops.dmx.quantize,
                (
                    fold,
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
