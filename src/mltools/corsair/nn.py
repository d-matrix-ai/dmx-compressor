from functools import reduce

from typing import Union, List, Tuple
from collections import OrderedDict
import sys
from types import SimpleNamespace
import torch
from torch import Tensor, Size
import torch.nn.functional as F
from mltools.numerical import (
    Format,
    NumericalCastMixin,
    Same,
    FixedPoint,
    FloatingPoint,
    BlockFloatingPoint,
    CastTo,
)
from mltools.sparse import (
    Sparseness,
    WeightSparseMixin,
    Dense,
    TopK,
    BlockTopK,
    Bernoulli,
    Sparsify,
)
from mltools.approximate import (
    ApproximationFunction,
    ApproximationMixin,
    NoApproximation,
    SoftmaxApproximation,
    GELUApproximation,
    LayerNormApproximation,
    Approximate,
    LowRankWeight,
)


class Cast(torch.nn.Module):
    r"""
    A container of Corsair-specific tensor numeric conversions,
    to be used as a stand-alone numerical conversion op by the user
    TODO: make this a true fake-quantization layer
    """

    def __init__(self, format="SAME") -> None:
        super().__init__()
        self.cast = CastTo(format=format)

    @property
    def format(self):
        return repr(self.cast.format)

    @format.setter
    def format(self, fmt):
        self.cast.set_format(fmt)

    def _transform(self, config):
        self.cast.format = Format.from_shorthand(config["format"])

    def _corsair_config(self, freeze=False):
        return CorsairModuleConfig.from_module(self, freeze)

    def forward(self, x):
        return self.cast(x)


class CorsairModule(
    ApproximationMixin, NumericalCastMixin, WeightSparseMixin, torch.nn.Module
):
    r"""
    Reimplemented torch.nn modules for Corsair
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _transform(self, config):
        # numerics transformation
        if "input_format" in config:
            self.input_cast.format = Format.from_shorthand(config["input_format"])
        if "output_format" in config:
            self.output_cast.format = Format.from_shorthand(config["output_format"])
        if self.accum_cast is not None and "accum_format" in config:
            self.accum_cast.format = Format.from_shorthand(config["accum_format"])
        if self.weight_cast is not None and "weight_format" in config:
            self.weight_cast.format = Format.from_shorthand(config["weight_format"])
        if self.bias_cast is not None and "bias_format" in config:
            self.bias_cast.format = Format.from_shorthand(config["bias_format"])
        # sparsity transformation
        if self.weight_sparsifier is not None and "weight_sparseness" in config:
            self.weight_sparsifier.configure(sparseness=config["weight_sparseness"])
        # custom logic transformation
        if "approximation_function" in config:
            self.approximator.function = ApproximationFunction.from_shorthand(
                config["approximation_function"]
            )

    def _corsair_config(self, freeze=False):
        return CorsairModuleConfig.from_module(self, freeze)

    @property
    def _weight(self):
        return (
            self.weight_cast(self.effective_weight)
            if self.weight_cast is not None
            else None
        )

    @property
    def _bias(self):
        return self.bias_cast(self.bias) if self.bias_cast is not None else None

    # def dmir_graph(self, node, omit_value=False):
    #     return dmir.Graph(
    #         name=dmir._make_var_name(
    #             name=node.name,
    #             suffix="wrap",
    #             end="",
    #         ),
    #         input=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="wrap_input"),
    #                 **dmir._tensor_meta_dict(node.args[0].meta["tensor_meta"]),
    #             ),
    #         ),
    #         intermediate=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="input"),
    #                 **dmir._tensor_meta_dict(node.args[0].meta["tensor_meta"]),
    #             ),
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="output"),
    #                 **dmir._tensor_meta_dict(node.meta["tensor_meta"]),
    #             ),
    #         ),
    #         output=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="wrap_output"),
    #                 **dmir._tensor_meta_dict(node.meta["tensor_meta"]),
    #             ),
    #         ),
    #         subgraph=(self._dmir_graph(node, omit_value=omit_value),),
    #         dependency=(
    #             dmir.Dependency(
    #                 operation="cast",
    #                 argument=[dmir._make_var_name(node.name, suffix="wrap_input")],
    #                 result=[dmir._make_var_name(node.name, suffix="input")],
    #                 attribute=dmir._corsair_specific_attributes(self.input_cast),
    #             ),
    #             dmir.Dependency(
    #                 operation=node.name,
    #                 argument=[dmir._make_var_name(node.name, suffix="input")],
    #                 result=[dmir._make_var_name(node.name, suffix="output")],
    #             ),
    #             dmir.Dependency(
    #                 operation="cast",
    #                 argument=[dmir._make_var_name(node.name, suffix="output")],
    #                 result=[dmir._make_var_name(node.name, suffix="wrap_output")],
    #                 attribute=dmir._corsair_specific_attributes(self.output_cast),
    #             ),
    #         ),
    #     )

    def forward(self, input: Tensor) -> Tensor:
        _input = self.input_cast(input)
        _output = self._forward(_input)
        output = self.output_cast(_output)
        return output


class CorsairModuleConfig(dict):
    @classmethod
    def from_module(cls, module: Union[CorsairModule, Cast], freeze=False):
        cc = SimpleNamespace()
        cc.instance = module.__class__.__name__
        if isinstance(module, Cast):
            if module.format is not None and (freeze or module.format != "SAME"):
                cc.format = module.format
        elif isinstance(module, CorsairModule):
            if module.input_format is not None and (
                freeze or module.input_format != "SAME"
            ):
                cc.input_format = module.input_format
            if module.output_format is not None and (
                freeze or module.output_format != "SAME"
            ):
                cc.output_format = module.output_format
            if module.accum_format is not None and (
                freeze or module.accum_format != "SAME"
            ):
                cc.accum_format = module.accum_format
            if module.weight_format is not None and (
                freeze or module.weight_format != "SAME"
            ):
                cc.weight_format = module.weight_format
            if module.bias_format is not None and (
                freeze or module.bias_format != "SAME"
            ):
                cc.bias_format = module.bias_format
            if module.weight_sparseness is not None and (
                freeze or module.weight_sparseness != "DENSE"
            ):
                cc.weight_sparseness = module.weight_sparseness
            if freeze or module.approximation_function != "NONE":
                cc.approximation_function = module.approximation_function
        return cc.__dict__


is_configurable = lambda m: isinstance(
    m,
    (
        CorsairModule,
        Cast,
    ),
)


class Linear(CorsairModule, torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, **kwargs)

    def _forward(self, _input: Tensor) -> Tensor:
        if isinstance(self.approximator.function, LowRankWeight):
            self.weight.data = self.approximator(self.weight.data)
        _product = self.accum_cast(torch.matmul(_input, self._weight.t()))
        if self.bias is not None:
            _output = torch.add(_product, self._bias)
        else:
            _output = _product
        return _output

    # def _dmir_graph(self, node, omit_value=False):
    #     return dmir.dump(
    #         self,
    #         *[
    #             torch.randn(
    #                 arg.meta["tensor_meta"].shape,
    #                 dtype=arg.meta["tensor_meta"].dtype,
    #                 device=self.weight.device,
    #             )
    #             for arg in node.args
    #         ],
    #         name=node.name,
    #         input_names=[dmir._make_var_name(n.__str__()) for n in node.args],
    #         output_names=[dmir._make_var_name(node.name)],
    #         omit_value=omit_value,
    #         metadata=dmir._nn_module_meta(self),
    # )


class Conv2d(CorsairModule, torch.nn.Conv2d):
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

    def _forward(self, _input: Tensor) -> Tensor:
        if isinstance(self.approximator.function, LowRankWeight):
            self.weight.data = self.approximator(self.weight.data)
        _convolution = self.accum_cast(self._conv_forward(_input, self._weight, None))
        if self.bias is not None:
            _output = torch.add(_convolution, self._bias.unsqueeze(-1).unsqueeze(-1))
        else:
            _output = _convolution
        return _output

    # def _dmir_graph(self, node, omit_value=False):
    #     return dmir.dump(
    #         self,
    #         *[
    #             torch.randn(
    #                 arg.meta["tensor_meta"].shape,
    #                 dtype=arg.meta["tensor_meta"].dtype,
    #                 device=self.weight.device,
    #             )
    #             for arg in node.args
    #         ],
    #         name=node.name,
    #         input_names=[dmir._make_var_name(n.__str__()) for n in node.args],
    #         output_names=[dmir._make_var_name(node.name)],
    #         omit_value=omit_value,
    #         metadata=dmir._nn_module_meta(self),
    #     )


class AdaptiveAvgPool2d(CorsairModule, torch.nn.AdaptiveAvgPool2d):
    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(_input)
        return _output

    # def _dmir_graph(self, node, omit_value=False):
    #     return dmir.Graph(
    #         name=node.name,
    #         input=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="input"),
    #                 **dmir._tensor_meta_dict(node.args[0].meta["tensor_meta"]),
    #             ),  # this is a dynamic input
    #         ),
    #         output=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="output"),
    #                 **dmir._tensor_meta_dict(node.meta["tensor_meta"]),
    #             ),
    #         ),
    #         dependency=(
    #             dmir.Dependency(
    #                 operation=f"adaptive_average_pool",
    #                 argument=(dmir._make_var_name(node.name, suffix="input"),),
    #                 result=(dmir._make_var_name(node.name, suffix="output"),),
    #                 attribute=(
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.INTS,
    #                         name="output_size",
    #                         integer_values=self.output_size,
    #                     )
    #                     if isinstance(self.output_size, tuple)
    #                     else dmir.Attribute(
    #                         kind=dmir.Attribute.INT,
    #                         name="output_size",
    #                         integer_value=self.output_size,
    #                     ),
    #                 ),
    #             ),
    #         ),
    #         metadata=dmir._nn_module_meta(self),
    #     )


class AvgPool2d(CorsairModule, torch.nn.AvgPool2d):
    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(_input)
        return _output

    # def _dmir_graph(self, node, omit_value=False):
    #     return dmir.Graph(
    #         name=node.name,
    #         input=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="input"),
    #                 **dmir._tensor_meta_dict(node.args[0].meta["tensor_meta"]),
    #             ),  # this is a dynamic input
    #         ),
    #         output=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="output"),
    #                 **dmir._tensor_meta_dict(node.meta["tensor_meta"]),
    #             ),
    #         ),
    #         dependency=(
    #             dmir.Dependency(
    #                 operation=f"average_pool",
    #                 argument=(dmir._make_var_name(node.name, suffix="input"),),
    #                 result=(dmir._make_var_name(node.name, suffix="output"),),
    #                 attribute=(
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.INTS,
    #                         name="kernel_size",
    #                         integer_values=self.kernel_size,
    #                     )
    #                     if isinstance(self.kernel_size, tuple)
    #                     else dmir.Attribute(
    #                         kind=dmir.Attribute.INT,
    #                         name="kernel_size",
    #                         integer_value=self.kernel_size,
    #                     ),
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.INTS,
    #                         name="stride",
    #                         integer_values=self.stride,
    #                     )
    #                     if isinstance(self.stride, tuple)
    #                     else dmir.Attribute(
    #                         kind=dmir.Attribute.INT,
    #                         name="stride",
    #                         integer_value=self.stride,
    #                     ),
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.INTS,
    #                         name="padding",
    #                         integer_values=self.padding,
    #                     )
    #                     if isinstance(self.padding, tuple)
    #                     else dmir.Attribute(
    #                         kind=dmir.Attribute.INT,
    #                         name="padding",
    #                         integer_value=self.padding,
    #                     ),
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.INTS,
    #                         name="dilation",
    #                         integer_values=self.dilation,
    #                     )
    #                     if isinstance(self.dilation, tuple)
    #                     else dmir.Attribute(
    #                         kind=dmir.Attribute.INT,
    #                         name="dilation",
    #                         integer_value=self.dilation,
    #                     ),
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.INT,
    #                         name="ceil_mode",
    #                         integer_value=int(self.ceil_mode),
    #                     ),
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.INT,
    #                         name="count_include_pad",
    #                         integer_value=int(self.count_include_pad),
    #                     ),
    #                 ),
    #             ),
    #         ),
    #         metadata=dmir._nn_module_meta(self),
    #     )


class MaxPool2d(CorsairModule, torch.nn.MaxPool2d):
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
        _output = self.approx_forward(_input)
        return _output

    # def _dmir_graph(self, node, omit_value=False):
    #     return dmir.Graph(
    #         name=node.name,
    #         input=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="input"),
    #                 **dmir._tensor_meta_dict(node.args[0].meta["tensor_meta"]),
    #             ),  # this is a dynamic input
    #         ),
    #         output=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="output"),
    #                 **dmir._tensor_meta_dict(node.meta["tensor_meta"]),
    #             ),
    #         ),
    #         dependency=(
    #             dmir.Dependency(
    #                 operation=f"max_pool",
    #                 argument=(dmir._make_var_name(node.name, suffix="input"),),
    #                 result=(dmir._make_var_name(node.name, suffix="output"),),
    #                 attribute=(
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.INTS,
    #                         name="kernel_size",
    #                         integer_values=self.kernel_size,
    #                     )
    #                     if isinstance(self.kernel_size, tuple)
    #                     else dmir.Attribute(
    #                         kind=dmir.Attribute.INT,
    #                         name="kernel_size",
    #                         integer_value=self.kernel_size,
    #                     ),
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.INTS,
    #                         name="stride",
    #                         integer_values=self.stride,
    #                     )
    #                     if isinstance(self.stride, tuple)
    #                     else dmir.Attribute(
    #                         kind=dmir.Attribute.INT,
    #                         name="stride",
    #                         integer_value=self.stride,
    #                     ),
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.INTS,
    #                         name="padding",
    #                         integer_values=self.padding,
    #                     )
    #                     if isinstance(self.padding, tuple)
    #                     else dmir.Attribute(
    #                         kind=dmir.Attribute.INT,
    #                         name="padding",
    #                         integer_value=self.padding,
    #                     ),
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.INTS,
    #                         name="dilation",
    #                         integer_values=self.dilation,
    #                     )
    #                     if isinstance(self.dilation, tuple)
    #                     else dmir.Attribute(
    #                         kind=dmir.Attribute.INT,
    #                         name="dilation",
    #                         integer_value=self.dilation,
    #                     ),
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.INT,
    #                         name="ceil_mode",
    #                         integer_value=int(self.ceil_mode),
    #                     ),
    #                 ),
    #             ),
    #         ),
    #         metadata=dmir._nn_module_meta(self),
    #     )


class Softmax(CorsairModule, torch.nn.Softmax):
    def __init__(self, dim: int = -1) -> None:
        super().__init__(dim=dim)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(_input, dim=self.dim)
        return _output

    # def _dmir_graph(self, node, omit_value=False):
    #     return dmir.Graph(
    #         name=node.name,
    #         input=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="input"),
    #                 **dmir._tensor_meta_dict(node.args[0].meta["tensor_meta"]),
    #             ),  # this is a dynamic input
    #         ),
    #         output=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="output"),
    #                 **dmir._tensor_meta_dict(node.meta["tensor_meta"]),
    #             ),
    #         ),
    #         dependency=(
    #             dmir.Dependency(
    #                 operation=f"softmax",
    #                 argument=(dmir._make_var_name(node.name, suffix="input"),),
    #                 result=(dmir._make_var_name(node.name, suffix="output"),),
    #                 attribute=[
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.INT,
    #                         name="dim",
    #                         integer_value=self.dim,
    #                     ),
    #                 ]
    #                 + dmir._corsair_specific_attributes(self),
    #             ),
    #         ),
    #         metadata=dmir._nn_module_meta(self),
    #     )


class LayerNorm(CorsairModule, torch.nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine
        )

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(
            _input, self.normalized_shape, self._weight, self._bias, self.eps
        )
        return _output

    # def _dmir_graph(self, node, omit_value=False):
    #     return dmir.Graph(
    #         name=node.name,
    #         input=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="input"),
    #                 **dmir._tensor_meta_dict(node.args[0].meta["tensor_meta"]),
    #             ),  # this is a dynamic input
    #         ),
    #         intermediate=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="weight"),
    #                 value=[]
    #                 if omit_value
    #                 else dmir._make_value_for_dumping(self.weight),
    #                 shape=self.weight.shape,
    #                 format=dmir._legal_format(self.weight.dtype),
    #             ),  # this is a static input
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="bias"),
    #                 value=[] if omit_value else dmir._make_value_for_dumping(self.bias),
    #                 shape=self.bias.shape,
    #                 format=dmir._legal_format(self.bias.dtype),
    #             ),  # this is a static input
    #         ),
    #         output=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="output"),
    #                 **dmir._tensor_meta_dict(node.meta["tensor_meta"]),
    #             ),
    #         ),
    #         dependency=(
    #             dmir.Dependency(
    #                 operation=f"{dmir._legal_op_type(node.graph._target_to_str(torch.layer_norm))}",
    #                 argument=(
    #                     dmir._make_var_name(node.name, suffix="input"),
    #                     dmir._make_var_name(node.name, suffix="weight"),
    #                     dmir._make_var_name(node.name, suffix="bias"),
    #                 ),
    #                 result=(dmir._make_var_name(node.name, suffix="output"),),
    #                 attribute=[
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.INTS,
    #                         name="normalized_shape",
    #                         integer_values=self.normalized_shape,
    #                     ),
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.FLOAT,
    #                         name="eps",
    #                         float_value=self.eps,
    #                     ),
    #                 ]
    #                 + dmir._corsair_specific_attributes(self),
    #             ),
    #         ),
    #         metadata=dmir._nn_module_meta(self),
    #     )


class BatchNorm2d(CorsairModule, torch.nn.BatchNorm2d):
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
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self._weight,
            self._bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        return _output

    # def _dmir_graph(self, node, omit_value=False):
    #     return dmir.Graph(
    #         name=node.name,
    #         input=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="input"),
    #                 **dmir._tensor_meta_dict(node.args[0].meta["tensor_meta"]),
    #             ),  # this is a dynamic input
    #         ),
    #         intermediate=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="running_mean"),
    #                 value=[]
    #                 if omit_value
    #                 else dmir._make_value_for_dumping(self.running_mean),
    #                 shape=self.running_mean.shape,
    #                 format=dmir._legal_format(self.running_mean.dtype),
    #             ),  # this is a static input
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="running_var"),
    #                 value=[]
    #                 if omit_value
    #                 else dmir._make_value_for_dumping(self.running_var),
    #                 shape=self.running_var.shape,
    #                 format=dmir._legal_format(self.running_var.dtype),
    #             ),  # this is a static input
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="weight"),
    #                 value=[]
    #                 if omit_value
    #                 else dmir._make_value_for_dumping(self.weight),
    #                 shape=self.weight.shape,
    #                 format=dmir._legal_format(self.weight.dtype),
    #             ),  # this is a static input
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="bias"),
    #                 value=[] if omit_value else dmir._make_value_for_dumping(self.bias),
    #                 shape=self.bias.shape,
    #                 format=dmir._legal_format(self.bias.dtype),
    #             ),  # this is a static input
    #         ),
    #         output=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="output"),
    #                 **dmir._tensor_meta_dict(node.meta["tensor_meta"]),
    #             ),
    #         ),
    #         dependency=(
    #             dmir.Dependency(
    #                 operation=f"{dmir._legal_op_type(node.graph._target_to_str(torch.batch_norm))}",
    #                 argument=(
    #                     dmir._make_var_name(node.name, suffix="input"),
    #                     dmir._make_var_name(node.name, suffix="running_mean"),
    #                     dmir._make_var_name(node.name, suffix="running_var"),
    #                     dmir._make_var_name(node.name, suffix="weight"),
    #                     dmir._make_var_name(node.name, suffix="bias"),
    #                 ),
    #                 result=(dmir._make_var_name(node.name, suffix="output"),),
    #                 attribute=(
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.FLOAT,
    #                         name="momentum",
    #                         float_value=self.momentum,
    #                     ),
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.FLOAT,
    #                         name="eps",
    #                         float_value=self.eps,
    #                     ),
    #                 ),
    #             ),
    #         ),
    #         metadata=dmir._nn_module_meta(self),
    #     )


class Dropout(CorsairModule, torch.nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p=p, inplace=inplace)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(_input)
        return _output

    # def _dmir_graph(self, node, omit_value=False):
    #     return dmir.Graph(
    #         name=node.name,
    #         input=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="input"),
    #                 **dmir._tensor_meta_dict(node.args[0].meta["tensor_meta"]),
    #             ),  # this is a dynamic input
    #         ),
    #         output=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="output"),
    #                 **dmir._tensor_meta_dict(node.meta["tensor_meta"]),
    #             ),
    #         ),
    #         dependency=(
    #             dmir.Dependency(
    #                 operation=f"dropout",
    #                 argument=(dmir._make_var_name(node.name, suffix="input"),),
    #                 result=(dmir._make_var_name(node.name, suffix="output"),),
    #                 attribute=(
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.FLOAT,
    #                         name="p",
    #                         float_value=self.p,
    #                     ),
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.INT,
    #                         name="inplace",
    #                         integer_value=int(self.inplace),
    #                     ),
    #                 ),
    #             ),
    #         ),
    #     )


class ReLU(CorsairModule, torch.nn.ReLU):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(_input)
        return _output

    # def _dmir_graph(self, node, omit_value=False):
    #     return dmir.Graph(
    #         name=node.name,
    #         input=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="input"),
    #                 **dmir._tensor_meta_dict(node.args[0].meta["tensor_meta"]),
    #             ),  # this is a dynamic input
    #         ),
    #         output=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="output"),
    #                 **dmir._tensor_meta_dict(node.meta["tensor_meta"]),
    #             ),
    #         ),
    #         dependency=(
    #             dmir.Dependency(
    #                 operation=f"relu",
    #                 argument=(dmir._make_var_name(node.name, suffix="input"),),
    #                 result=(dmir._make_var_name(node.name, suffix="output"),),
    #                 attribute=(
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.INT,
    #                         name="inplace",
    #                         integer_value=int(self.inplace),
    #                     ),
    #                 ),
    #             ),
    #         ),
    #     )


class ReLU6(CorsairModule, torch.nn.ReLU6):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(_input)
        return _output

    # def _dmir_graph(self, node, omit_value=False):
    #     return dmir.Graph(
    #         name=node.name,
    #         input=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="input"),
    #                 **dmir._tensor_meta_dict(node.args[0].meta["tensor_meta"]),
    #             ),  # this is a dynamic input
    #         ),
    #         output=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="output"),
    #                 **dmir._tensor_meta_dict(node.meta["tensor_meta"]),
    #             ),
    #         ),
    #         dependency=(
    #             dmir.Dependency(
    #                 operation=f"relu6",
    #                 argument=(dmir._make_var_name(node.name, suffix="input"),),
    #                 result=(dmir._make_var_name(node.name, suffix="output"),),
    #                 attribute=(
    #                     dmir.Attribute(
    #                         kind=dmir.Attribute.INT,
    #                         name="inplace",
    #                         integer_value=int(self.inplace),
    #                     ),
    #                 ),
    #             ),
    #         ),
    #     )


class Tanh(CorsairModule, torch.nn.Tanh):
    def __init__(self) -> None:
        super().__init__()

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(_input)
        return _output

    # def _dmir_graph(self, node, omit_value=False):
    #     return dmir.Graph(
    #         name=node.name,
    #         input=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="input"),
    #                 **dmir._tensor_meta_dict(node.args[0].meta["tensor_meta"]),
    #             ),  # this is a dynamic input
    #         ),
    #         output=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="output"),
    #                 **dmir._tensor_meta_dict(node.meta["tensor_meta"]),
    #             ),
    #         ),
    #         dependency=(
    #             dmir.Dependency(
    #                 operation=f"tanh",
    #                 argument=(dmir._make_var_name(node.name, suffix="input"),),
    #                 result=(dmir._make_var_name(node.name, suffix="output"),),
    #                 attribute=dmir._corsair_specific_attributes(self),
    #             ),
    #         ),
    #         metadata=dmir._nn_module_meta(self),
    #     )


class GELU(CorsairModule, torch.nn.GELU):
    def __init__(self) -> None:
        super().__init__()

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(_input)
        return _output

    # def _dmir_graph(self, node, omit_value=False):
    #     return dmir.Graph(
    #         name=node.name,
    #         input=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="input"),
    #                 **dmir._tensor_meta_dict(node.args[0].meta["tensor_meta"]),
    #             ),  # this is a dynamic input
    #         ),
    #         output=(
    #             dmir.Tensor(
    #                 name=dmir._make_var_name(node.name, suffix="output"),
    #                 **dmir._tensor_meta_dict(node.meta["tensor_meta"]),
    #             ),
    #         ),
    #         dependency=(
    #             dmir.Dependency(
    #                 operation=f"gelu",
    #                 argument=(dmir._make_var_name(node.name, suffix="input"),),
    #                 result=(dmir._make_var_name(node.name, suffix="output"),),
    #                 attribute=dmir._corsair_specific_attributes(self),
    #             ),
    #         ),
    #         metadata=dmir._nn_module_meta(self),
    #     )
