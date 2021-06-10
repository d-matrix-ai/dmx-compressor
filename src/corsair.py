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
    WeightSparseMixin,
    Dense,
    TopK,
    BlockTopK,
    Bernoulli,
    Sparsify,
)
from torch.nn.modules.pooling import MaxPool2d
from utils import load_config_file

__ALL__ = ["nn", "CorsairModule"]


class CorsairModule(torch.nn.Module):
    r"""
    Model container equipped with corsair transform
    """

    def __init__(self):
        super().__init__()

    def transform(self, config_file="configs/corsair.yaml"):
        r"""
        Model conversion for Corsair numerics/sparsity simulation/optimization
        """
        config = load_config_file(config_file)

        for n, m in self.named_modules():
            for r in config["transformation_rules"]:
                if (
                    isinstance(m, getattr(sys.modules[__name__], r["instance"]))
                    and all([_n in n for _n in r["name_includes"]])
                    and all([not _n in n for _n in r["name_excludes"]])
                ):
                    m.transform(r["config"])

    def load_state_dict(
        self, state_dict: "OrderedDict[str, Tensor]", strict: bool = False
    ):
        return super().load_state_dict(state_dict, strict=strict)


class CorsairMixin(BoundaryCastMixin, WeightSparseMixin):
    r"""
    Extending torch.nn.Module
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def transform(self, config):
        # numerics transformation
        self.input_cast.format = Format.str2format(config["input_format"])
        self.output_cast.format = Format.str2format(config["output_format"])
        if self.accum_cast is not None:
            self.accum_cast.format = Format.str2format(config["accum_format"])
        if self.weight_cast is not None:
            self.weight_cast.format = Format.str2format(config["weight_format"])
        if self.bias_cast is not None:
            self.bias_cast.format = Format.str2format(config["bias_format"])
        # sparsity transformation
        if self.weight_sparsifier is not None:
            self.weight_sparsifier.sparseness = Sparseness.str2sparseness(
                config["weight_sparseness"]
            )
            ### TODO: need to figure out a better way of handling score setting
            self.weight_sparsifier.set_score(torch.abs(self.weight))
            ###


class Linear(CorsairMixin, torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, input: Tensor) -> Tensor:
        _input = self.input_cast(input)
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
        output = self.output_cast(_output)
        return output


class Conv2d(CorsairMixin, torch.nn.Conv2d):
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

    def forward(self, input: Tensor) -> Tensor:
        _input = self.input_cast(input)
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
        output = self.output_cast(_output)
        return output


class AdaptiveAvgPool2d(CorsairMixin, torch.nn.AdaptiveAvgPool2d):
    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def forward(self, input: Tensor) -> Tensor:
        _input = self.input_cast(input)
        _output = super().forward(_input)
        output = self.output_cast(_output)
        return output


class MaxPool2d(CorsairMixin, torch.nn.MaxPool2d):
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

    def forward(self, input: Tensor) -> Tensor:
        _input = self.input_cast(input)
        _output = super().forward(_input)
        output = self.output_cast(_output)
        return output


class Softmax(CorsairMixin, torch.nn.Softmax):
    def __init__(self, dim: int = -1) -> None:
        super().__init__(dim=dim)

    def forward(self, input: Tensor) -> Tensor:
        _output = self.input_cast(input)
        _output = super().forward(_output)
        output = self.output_cast(_output)
        return output


class LayerNorm(CorsairMixin, torch.nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine
        )

    def forward(self, input: Tensor) -> Tensor:
        _input = self.input_cast(input)
        _weight = self.weight_cast(self.weight)
        _bias = self.bias_cast(self.bias)
        _output = F.layer_norm(_input, self.normalized_shape, _weight, _bias, self.eps)
        output = self.output_cast(_output)
        return output


class BatchNorm2d(CorsairMixin, torch.nn.BatchNorm2d):
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

    def forward(self, input: Tensor) -> Tensor:
        _input = self.input_cast(input)
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
        output = self.output_cast(_output)
        return output


class Dropout(CorsairMixin, torch.nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p=p, inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        _output = self.input_cast(input)
        _output = super().forward(_output)
        output = self.output_cast(_output)
        return output


class ReLU(CorsairMixin, torch.nn.ReLU):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        _output = self.input_cast(input)
        _output = super().forward(_output)
        output = self.output_cast(_output)
        return output


class ReLU6(CorsairMixin, torch.nn.ReLU6):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        _output = self.input_cast(input)
        _output = super().forward(_output)
        output = self.output_cast(_output)
        return output


class Tanh(CorsairMixin, torch.nn.Tanh):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        _output = self.input_cast(input)
        _output = super().forward(_output)
        output = self.output_cast(_output)
        return output


# overload torch.nn modules
nn = torch.nn
nn.Module = CorsairModule
nn.Linear = Linear
nn.Conv2d = Conv2d
nn.AdaptiveAvgPool2d = AdaptiveAvgPool2d
nn.MaxPool2d = MaxPool2d
nn.BatchNorm2d = BatchNorm2d
nn.LayerNorm = LayerNorm
nn.Dropout = Dropout
nn.Softmax = Softmax
nn.ReLU = ReLU
nn.ReLU6 = ReLU6
nn.Tanh = Tanh

if __name__ == "__main__":
    pass
