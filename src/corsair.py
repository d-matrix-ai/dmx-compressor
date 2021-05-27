import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from numerical import FixedPoint, FloatingPoint, BlockFloatingPoint, CastTo
from sparse import WeightSparseMixin
# from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize


__ALL__ = ["Linear",]

# TODO: class CorsairConfig 
# Corsair configuration
IMC_INPUT_FORMAT = BlockFloatingPoint(
    precision=8,
    block_size=64,
    rounding="nearest",
)
IMC_WEIGHT_FORMAT = BlockFloatingPoint(
    precision=8,
    block_size=64,
    rounding="nearest",
)
IMC_OUTPUT_FORMAT = FloatingPoint()
OB_FORMAT = FloatingPoint()
SIMD_FORMAT = FloatingPoint()  # FixedPoint(
#     precision=25,
#     fraction=12,
#     symmetric=True,
#     rounding="nearest"
# )


class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.input_cast = CastTo(format=IMC_INPUT_FORMAT)
        self.weight_cast = CastTo(format=IMC_WEIGHT_FORMAT)
        self.product_cast = CastTo(format=IMC_OUTPUT_FORMAT, enabled=False)
        self.bias_cast = CastTo(format=OB_FORMAT, enabled=False)
        self.output_cast = CastTo(format=OB_FORMAT, enabled=False)

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

    # def extra_repr(self) -> str:
    #     return super().extra_repr() + "\nCorsair simulation"