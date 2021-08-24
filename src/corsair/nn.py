from functools import reduce
from typing import Union, List, Tuple
from collections import OrderedDict
import sys
import torch
from torch import Tensor, Size
import torch.nn.functional as F
from numerical import (
    Format,
    BoundaryCastMixin,
    Same,
    FixedPoint,
    FloatingPoint,
    BlockFloatingPoint,
    CastTo,
)
from sparse import (
    Sparseness,
    Sparsifier,
    WeightSparseMixin,
    Dense,
    TopK,
    BlockTopK,
    Bernoulli,
    Sparsify,
)
from approximate import (
    ApproximationFunction,
    ApproximationMixin,
    NoApproximation,
    SoftmaxApproximation,
    GELUApproximation,
    LayerNormApproximation,
    Approximate,
)
from utils import load_config_file


class CorsairModule(
    ApproximationMixin, BoundaryCastMixin, WeightSparseMixin, torch.nn.Module
):
    r"""
    Container equipped with corsair transform, extending torch.nn.Module
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dmir = None

    def transform(self, config="configs/corsair.yaml"):
        r"""
        Model conversion for Corsair numerics/sparsity simulation/optimization
        """
        if isinstance(config, str):
            config = load_config_file(config)

        for n, m in self.named_modules():
            for r in config["transformation_rules"]:
                if (
                    isinstance(m, getattr(sys.modules[__name__], r["instance"]))
                    and all([_n in n for _n in r["name_includes"]])
                    and all([not _n in n for _n in r["name_excludes"]])
                ):
                    m._transform(r["config"])

    def load_state_dict(
        self, state_dict: "OrderedDict[str, Tensor]", strict: bool = False
    ):
        return super().load_state_dict(state_dict, strict=strict)

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
            self.weight_sparsifier.sparseness = Sparseness.from_shorthand(
                config["weight_sparseness"]
            )
            ### TODO: need to figure out a better way of handling score setting
            self.weight_sparsifier.set_score(torch.abs(self.weight))
            ###
        # custom logic transformation
        if "approximation_function" in config:
            self.approximator.function = ApproximationFunction.from_shorthand(
                config["approximation_function"]
            )

    def forward(self, input: Tensor) -> Tensor:
        _input = self.input_cast(input)
        _output = self._forward(_input)
        output = self.output_cast(_output)
        return output


class Linear(CorsairModule, torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)

    def _forward(self, _input: Tensor) -> Tensor:
        _weight = self.weight_cast(self.effective_weight)
        if isinstance(self.accum_cast.format, BlockFloatingPoint):
            B_i = (
                self.input_cast.format.block_size
                if isinstance(self.input_cast.format, BlockFloatingPoint)
                else 1
            )
            B_w = (
                self.weight_cast.format.block_size
                if isinstance(self.weight_cast.format, BlockFloatingPoint)
                else 1
            )
            B = max(64, min(B_i, B_w))
            _inputs = torch.split(_input, B, dim=-1)
            _weights = torch.split(_weight, B, dim=-1)
            _products = (
                self.accum_cast(F.linear(_i, _w, None))
                for _i, _w in zip(_inputs, _weights)
            )
            _product = reduce((lambda x, y: self.accum_cast(x + y)), _products)
        else:
            _product = self.accum_cast(F.linear(_input, _weight, None))
        if self.bias is not None:
            _bias = self.bias_cast(self.bias)
            _output = torch.add(_product, _bias)
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
        )

    def _forward(self, _input: Tensor) -> Tensor:
        _weight = self.weight_cast(self.effective_weight)
        if isinstance(self.accum_cast.format, BlockFloatingPoint):
            B_i = (
                self.input_cast.format.block_size
                if isinstance(self.input_cast.format, BlockFloatingPoint)
                else 1
            )
            B_w = (
                self.weight_cast.format.block_size
                if isinstance(self.weight_cast.format, BlockFloatingPoint)
                else 1
            )
            B = max(64, min(B_i, B_w), self.groups)
            _inputs = torch.split(_input, B, dim=1)
            _weights = torch.split(_weight, B, dim=1)
            _convolutions = (
                self.accum_cast(self._conv_forward(_i, _w))
                for _i, _w in zip(_inputs, _weights)
            )
            _convolution = reduce((lambda x, y: self.accum_cast(x + y)), _convolutions)
        else:
            _convolution = self.accum_cast(self._conv_forward(_input, _weight))
        if self.bias is not None:
            _bias = self.bias_cast(self.bias)
            _output = torch.add(_convolution, _bias.unsqueeze(-1).unsqueeze(-1))
        else:
            _output = _convolution
        return _output


class AdaptiveAvgPool2d(CorsairModule, torch.nn.AdaptiveAvgPool2d):
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
        _weight = self.weight_cast(self.weight)
        _bias = self.bias_cast(self.bias)
        _output = F.layer_norm(_input, self.normalized_shape, _weight, _bias, self.eps)
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
        _weight = self.weight_cast(self.weight)
        _bias = self.bias_cast(self.bias)
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
            _weight,
            _bias,
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

