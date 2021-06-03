from typing import Union, List, Tuple
import torch
from torch import Tensor, Size
import torch.nn.functional as F
from numerical import (
    BoundaryCastMixin,
    Same,
    FixedPoint,
    FloatingPoint,
    BlockFloatingPoint,
    CastTo,
)
from sparse import WeightSparseMixin


__ALL__ = ["nn", "transform", "CorsairConfig"]


class CorsairConfig:
    DUMMY_FORMAT = Same()
    IMC_GEMM_INPUT_FORMAT_HIGH = BlockFloatingPoint(
        precision=8,
        block_size=64,
        block_dim=-1,
        rounding="nearest",
    )
    IMC_GEMM_INPUT_FORMAT_LOW = BlockFloatingPoint(
        precision=4,
        block_size=128,
        block_dim=-1,
        rounding="nearest",
    )
    IMC_CONV_INPUT_FORMAT_HIGH = BlockFloatingPoint(
        precision=8,
        block_size=64,
        block_dim=1,
        rounding="nearest",
    )
    IMC_CONV_INPUT_FORMAT_LOW = BlockFloatingPoint(
        precision=4,
        block_size=128,
        block_dim=1,
        rounding="nearest",
    )
    IMC_ACCUM_FORMAT = FloatingPoint()
    IMC_OUTPUT_FORMAT = FloatingPoint()
    OB_FORMAT = FloatingPoint()
    SIMD_FORMAT = FloatingPoint()  # FixedPoint(
    #     precision=25,
    #     fraction=12,
    #     symmetric=True,
    #     rounding="nearest",
    # )
    # WEIGHT_SPARSITY = Sparsity()


class CorsairModule(BoundaryCastMixin, WeightSparseMixin):
    r""" """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.init_casts()

    def transform(self, config):
        # numerics transformation
        self.input_cast.format = config["input_format"]
        self.output_cast.format = config["output_format"]
        if self.accum_cast is not None:
            self.accum_cast.format = config["accum_format"]
        if self.weight_cast is not None:
            self.weight_cast.format = config["weight_format"]
        if self.bias_cast is not None:
            self.bias_cast.format = config["bias_format"]
        # sparsity transformation
        # TODO


class Linear(CorsairModule, torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, input: Tensor) -> Tensor:
        _input = self.input_cast(input)
        _weight = self.weight_cast(self.weight)
        _product = self.accum_cast(F.linear(_input, _weight, None))
        _bias = self.bias_cast(self.bias)
        _output = torch.add(_product, _bias)
        output = self.output_cast(_output)
        return output


class Softmax(CorsairModule, torch.nn.Softmax):
    def __init__(self, dim: int = -1) -> None:
        super().__init__(dim=dim)

    def forward(self, input: Tensor) -> Tensor:
        _output = self.input_cast(input)
        _output = super().forward(_output)
        output = self.output_cast(_output)
        return output


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

    def forward(self, input: Tensor) -> Tensor:
        _input = self.input_cast(input)
        _weight = self.weight_cast(self.weight)
        _bias = self.bias_cast(self.bias)
        output = F.layer_norm(_input, self.normalized_shape, _weight, _bias, self.eps)
        return output


class Dropout(CorsairModule, torch.nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p=p, inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        _output = self.input_cast(input)
        _output = super().forward(_output)
        output = self.output_cast(_output)
        return output


# overload torch.nn modules
nn = torch.nn
nn.Linear = Linear
nn.Softmax = Softmax
nn.LayerNorm = LayerNorm
nn.Dropout = Dropout


def transform(model):
    r"""
    Quick and dirty model conversion for Corsair numerical simulation/optimization
    TODO: general transformation API with yaml config and regex pattern matching
    """
    config_qkv = dict(
        input_format=CorsairConfig.IMC_GEMM_INPUT_FORMAT_LOW,
        output_format=CorsairConfig.IMC_GEMM_INPUT_FORMAT_LOW,
        accum_format=CorsairConfig.DUMMY_FORMAT,
        weight_format=CorsairConfig.IMC_GEMM_INPUT_FORMAT_LOW,
        bias_format=CorsairConfig.DUMMY_FORMAT,
    )
    config_dense = dict(
        input_format=CorsairConfig.IMC_GEMM_INPUT_FORMAT_LOW,
        output_format=CorsairConfig.DUMMY_FORMAT,
        accum_format=CorsairConfig.DUMMY_FORMAT,
        weight_format=CorsairConfig.IMC_GEMM_INPUT_FORMAT_LOW,
        bias_format=CorsairConfig.DUMMY_FORMAT,
    )
    config_do = dict(
        input_format=CorsairConfig.DUMMY_FORMAT,
        output_format=CorsairConfig.IMC_GEMM_INPUT_FORMAT_LOW,
    )
    for n, m in model.named_modules():
        if isinstance(m, Linear):
            if ".encoder.layer" in n and "attention.self.query" in n:
                m.transform(
                    config_qkv
                )  # for torch.matmul(query_layer, key_layer.transpose(-1, -2))
            elif ".encoder.layer" in n and "attention.self.key" in n:
                m.transform(
                    config_qkv
                )  # for torch.matmul(query_layer, key_layer.transpose(-1, -2))
            elif ".encoder.layer" in n and "attention.self.value" in n:
                m.transform(
                    config_qkv
                )  # for torch.matmul(attention_probs, value_layer)
            elif ".encoder.layer" in n and "attention.output.dense" in n:
                m.transform(config_dense)
            elif ".encoder.layer" in n and "intermediate.dense" in n:
                m.transform(config_dense)
            elif ".encoder.layer" in n and "output.dense" in n and not "attention" in n:
                m.transform(config_dense)
        elif isinstance(m, Dropout):
            if ".encoder.layer" in n and "attention.self.dropout" in n:
                m.transform(config_do)  # for torch.matmul(attention_probs, value_layer)
    for m in model.modules():
        if isinstance(m, LayerNorm):
            pass
    print(model)
    return model
