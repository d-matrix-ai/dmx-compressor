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
        _product = self.accum_cast(
            torch.matmul(_input.to(self._weight.dtype), self._weight.t())
        )
        if self.bias is not None:
            _output = torch.add(_product, self._bias)
        else:
            _output = _product
        return _output


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
        _convolution = self.accum_cast(
            self._conv_forward(_input.to(self._weight.dtype), self._weight, None)
        )
        if self.bias is not None:
            _output = torch.add(_convolution, self._bias.unsqueeze(-1).unsqueeze(-1))
        else:
            _output = _convolution
        return _output


class AdaptiveAvgPool2d(CorsairModule, torch.nn.AdaptiveAvgPool2d):
    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(_input)
        return _output


class AvgPool2d(CorsairModule, torch.nn.AvgPool2d):
    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(_input)
        return _output


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


class Softmax(CorsairModule, torch.nn.Softmax):
    def __init__(self, dim: int = -1) -> None:
        super().__init__(dim=dim)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(_input, dim=self.dim)
        return _output


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


class Dropout(CorsairModule, torch.nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p=p, inplace=inplace)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(_input)
        return _output


class ReLU(CorsairModule, torch.nn.ReLU):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(_input)
        return _output


class ReLU6(CorsairModule, torch.nn.ReLU6):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(_input)
        return _output


class Tanh(CorsairModule, torch.nn.Tanh):
    def __init__(self) -> None:
        super().__init__()

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(_input)
        return _output


class GELU(CorsairModule, torch.nn.GELU):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(_input)
        return _output
