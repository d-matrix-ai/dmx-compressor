import math
from collections import UserDict
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from numerical import Same, FixedPoint, FloatingPoint, BlockFloatingPoint, CastTo
# from sparse import WeightSparseMixin


__ALL__ = ["Linear",]


class CorsairConfig:
    IMC_INPUT_FORMAT_HIGH = BlockFloatingPoint(
        precision=8,
        block_size=64,
        rounding="nearest",
    )
    IMC_WEIGHT_FORMAT_HIGH = BlockFloatingPoint(
        precision=8,
        block_size=64,
        rounding="nearest",
    )
    IMC_INPUT_FORMAT_LOW = BlockFloatingPoint(
        precision=4,
        block_size=128,
        rounding="nearest",
    )
    IMC_WEIGHT_FORMAT_LOW = BlockFloatingPoint(
        precision=4,
        block_size=128,
        rounding="nearest",
    )
    IMC_ACCUM_FORMAT = FloatingPoint()
    IMC_OUTPUT_FORMAT = FloatingPoint()
    OB_FORMAT = FloatingPoint()
    SIMD_FORMAT = FixedPoint(
        precision=25,
        fraction=12,
        symmetric=True,
        rounding="nearest"
    )

class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.input_cast = CastTo()
        self.weight_cast = CastTo()
        self.product_cast = CastTo()
        self.bias_cast = CastTo()
        self.output_cast = CastTo()

    @classmethod
    def from_existing(cls, obj):
        pass

    def forward(self, input: Tensor) -> Tensor:
        _input = self.input_cast(input)
        _weight = self.weight_cast(self.weight)
        _product = self.product_cast(
            F.linear(_input, _weight, None)
        )
        _bias = self.bias_cast(self.bias)
        _output = torch.add(_product, _bias)
        output = self.output_cast(_output)
        return output
