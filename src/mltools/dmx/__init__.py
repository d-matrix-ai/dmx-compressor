from types import SimpleNamespace
from . import nn
from ..numerical import CastTo, Format
from ..sparse import Sparsify, Sparseness
from ..functional import Approximate, ApproximationFunction
from .model import (
    DmxModel,
    DmxConfig,
    DmxConfigRule,
    DmxPipelineMixin,
    DmxSimplePipeline,
    Model,
)
from .hf import pipeline, dmx_transform
from .nn import *

# Numerical format aliases
format = SimpleNamespace(
    FLOAT32=Format.from_shorthand("FP[1|8|23,127](_N)"),
    FLOAT16=Format.from_shorthand("FP[1|5|10,15](FN)"),
    BFLOAT16=Format.from_shorthand("FP[1|8|7,127](FN)"),
    AFLOAT8=Format.from_shorthand("FP[1|4|3,7](_N)"),
    BFLOAT8=Format.from_shorthand("FP[1|5|2,15](_N)"),
    INT8=Format.from_shorthand("XP[8,0](CSN)"),
    INT4=Format.from_shorthand("XP[4,0](CSN)"),
    BFP32_1=Format.from_shorthand("BFP[24|8]{1}(SN)"),
    BFP24_64=Format.from_shorthand("BFP[16|8]{64}(SN)"),
    # BFP24_32_LD=Format.from_shorthand("BFP[16|8]{32,-1}(SN)"),
    # BFP24_32_FD=Format.from_shorthand("BFP[16|8]{32,1}(SN)"),
    # BFP24_16_LD=Format.from_shorthand("BFP[16|8]{16,-1}(SN)"),
    # BFP24_16_FD=Format.from_shorthand("BFP[16|8]{16,1}(SN)"),
    # BFP16_128_LD=Format.from_shorthand("BFP[8|8]{128,-1}(SN)"),
    # BFP16_128_FD=Format.from_shorthand("BFP[8|8]{128,1}(SN)"),
    BFP16_64=Format.from_shorthand("BFP[8|8]{64}(SN)"),
    # BFP16_64_PD=Format.from_shorthand("BFP[8|8]{64,-2}(SN)"),
    # BFP16_64_FD=Format.from_shorthand("BFP[8|8]{64,1}(SN)"),
    # BFP16_32_LD=Format.from_shorthand("BFP[8|8]{32,-1}(SN)"),
    # BFP16_32_FD=Format.from_shorthand("BFP[8|8]{32,1}(SN)"),
    # BFP16_16_LD=Format.from_shorthand("BFP[8|8]{16,-1}(SN)"),
    # BFP16_16_FD=Format.from_shorthand("BFP[8|8]{16,1}(SN)"),
    # BFP14_128_LD=Format.from_shorthand("BFP[6|8]{128,-1}(SN)"),
    # BFP14_128_FD=Format.from_shorthand("BFP[6|8]{128,1}(SN)"),
    # BFP14_64_LD=Format.from_shorthand("BFP[6|8]{64,-1}(SN)"),
    # BFP14_64_FD=Format.from_shorthand("BFP[6|8]{64,1}(SN)"),
    # BFP14_32_LD=Format.from_shorthand("BFP[6|8]{32,-1}(SN)"),
    # BFP14_32_FD=Format.from_shorthand("BFP[6|8]{32,1}(SN)"),
    # BFP14_16_LD=Format.from_shorthand("BFP[6|8]{16,-1}(SN)"),
    # BFP14_16_FD=Format.from_shorthand("BFP[6|8]{16,1}(SN)"),
    # BFP12_128_LD=Format.from_shorthand("BFP[4|8]{128,-1}(SN)"),
    # BFP12_128_FD=Format.from_shorthand("BFP[4|8]{128,1}(SN)"),
    # BFP12_64_LD=Format.from_shorthand("BFP[4|8]{64,-1}(SN)"),
    # BFP12_64_FD=Format.from_shorthand("BFP[4|8]{64,1}(SN)"),
    # BFP12_32_LD=Format.from_shorthand("BFP[4|8]{32,-1}(SN)"),
    # BFP12_32_FD=Format.from_shorthand("BFP[4|8]{32,1}(SN)"),
    # BFP12_16_LD=Format.from_shorthand("BFP[4|8]{16,-1}(SN)"),
    # BFP12_16_FD=Format.from_shorthand("BFP[4|8]{16,1}(SN)"),
    # BFP16A_128_LD=Format.from_shorthand("BFP[8|8]{128,-1}(_N)"),
    # BFP16A_128_FD=Format.from_shorthand("BFP[8|8]{128,1}(_N)"),
    # BFP16A_64_LD=Format.from_shorthand("BFP[8|8]{64,-1}(_N)"),
    # BFP16A_64_FD=Format.from_shorthand("BFP[8|8]{64,1}(_N)"),
    # BFP16A_32_LD=Format.from_shorthand("BFP[8|8]{32,-1}(_N)"),
    # BFP16A_32_FD=Format.from_shorthand("BFP[8|8]{32,1}(_N)"),
    # BFP16A_16_LD=Format.from_shorthand("BFP[8|8]{16,-1}(_N)"),
    # BFP16A_16_FD=Format.from_shorthand("BFP[6|8]{16,1}(_N)"),
    # BFP14A_128_LD=Format.from_shorthand("BFP[6|8]{128,-1}(_N)"),
    # BFP14A_128_FD=Format.from_shorthand("BFP[6|8]{128,1}(_N)"),
    # BFP14A_64_LD=Format.from_shorthand("BFP[6|8]{64,-1}(_N)"),
    # BFP14A_64_FD=Format.from_shorthand("BFP[6|8]{64,1}(_N)"),
    # BFP14A_32_LD=Format.from_shorthand("BFP[6|8]{32,-1}(_N)"),
    # BFP14A_32_FD=Format.from_shorthand("BFP[6|8]{32,1}(_N)"),
    # BFP14A_16_LD=Format.from_shorthand("BFP[6|8]{16,-1}(_N)"),
    # BFP14A_16_FD=Format.from_shorthand("BFP[6|8]{16,1}(_N)"),
    # BFP12A_128_LD=Format.from_shorthand("BFP[4|8]{128,-1}(_N)"),
    # BFP12A_128_FD=Format.from_shorthand("BFP[4|8]{128,1}(_N)"),
    # BFP12A_64_LD=Format.from_shorthand("BFP[4|8]{64,-1}(_N)"),
    # BFP12A_64_FD=Format.from_shorthand("BFP[4|8]{64,1}(_N)"),
    # BFP12A_32_LD=Format.from_shorthand("BFP[4|8]{32,-1}(_N)"),
    # BFP12A_32_FD=Format.from_shorthand("BFP[4|8]{32,1}(_N)"),
    # BFP12A_16_LD=Format.from_shorthand("BFP[4|8]{16,-1}(_N)"),
    # BFP12A_16_FD=Format.from_shorthand("BFP[4|8]{16,1}(_N)"),
    SBFP12_16_4_LD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,4](FN)>{16,-1}"),
    SBFP12_16_4_FD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,4](FN)>{16,1}"),
    SBFP12_16_5_LD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,5](FN)>{16,-1}"),
    SBFP12_16_5_FD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,5](FN)>{16,1}"),
    SBFP12_16_6_LD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,6](FN)>{16,-1}"),
    SBFP12_16_6_FD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,6](FN)>{16,1}"),
    SBFP12_16_7_LD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,7](FN)>{16,-1}"),
    SBFP12_16_7_FD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,7](FN)>{16,1}"),
    SBFP12_16_8_LD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,8](FN)>{16,-1}"),
    SBFP12_16_8_FD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,8](FN)>{16,1}"),
    SBFP12_16_9_LD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,9](FN)>{16,-1}"),
    SBFP12_16_9_FD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,9](FN)>{16,1}"),
    SBFP12_16_10_LD=Format.from_shorthand(
        "SBFP<XP[4,0](CSN)><FP[0|4|4,10](FN)>{16,-1}"
    ),
    SBFP12_16_10_FD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,10](FN)>{16,1}"),
    SBFP12_16_11_LD=Format.from_shorthand(
        "SBFP<XP[4,0](CSN)><FP[0|4|4,11](FN)>{16,-1}"
    ),
    SBFP12_16_11_FD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,11](FN)>{16,1}"),
    SBFP12_16_12_LD=Format.from_shorthand(
        "SBFP<XP[4,0](CSN)><FP[0|4|4,12](FN)>{16,-1}"
    ),
    SBFP12_16_12_FD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,12](FN)>{16,1}"),
    SBFP12_16_13_LD=Format.from_shorthand(
        "SBFP<XP[4,0](CSN)><FP[0|4|4,13](FN)>{16,-1}"
    ),
    SBFP12_16_13_FD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,13](FN)>{16,1}"),
    SBFP12_16_14_LD=Format.from_shorthand(
        "SBFP<XP[4,0](CSN)><FP[0|4|4,14](FN)>{16,-1}"
    ),
    SBFP12_16_14_FD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,14](FN)>{16,1}"),
    SBFP12_16_15_LD=Format.from_shorthand(
        "SBFP<XP[4,0](CSN)><FP[0|4|4,15](FN)>{16,-1}"
    ),
    SBFP12_16_15_FD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,15](FN)>{16,1}"),
    SBFP12_16_16_LD=Format.from_shorthand(
        "SBFP<XP[4,0](CSN)><FP[0|4|4,16](FN)>{16,-1}"
    ),
    SBFP12_16_16_FD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,16](FN)>{16,1}"),
    SBFP12_16_17_LD=Format.from_shorthand(
        "SBFP<XP[4,0](CSN)><FP[0|4|4,17](FN)>{16,-1}"
    ),
    SBFP12_16_17_FD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,17](FN)>{16,1}"),
    SBFP12_16_18_LD=Format.from_shorthand(
        "SBFP<XP[4,0](CSN)><FP[0|4|4,18](FN)>{16,-1}"
    ),
    SBFP12_16_18_FD=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,18](FN)>{16,1}"),
    MXFP8_E4M3K128_LD=Format.from_shorthand("MXFP8[E4M3]{128,-1}"),
    MXFP8_E4M3K128_FD=Format.from_shorthand("MXFP8[E4M3]{128,1}"),
    MXFP8_E4M3K64_LD=Format.from_shorthand("MXFP8[E4M3]{64,-1}"),
    MXFP8_E4M3K64_FD=Format.from_shorthand("MXFP8[E4M3]{64,1}"),
    MXFP8_E4M3K32_LD=Format.from_shorthand("MXFP8[E4M3]{32,-1}"),
    MXFP8_E4M3K32_FD=Format.from_shorthand("MXFP8[E4M3]{32,1}"),
    MXFP8_E5M2K128_LD=Format.from_shorthand("MXFP8[E5M2]{128,-1}"),
    MXFP8_E5M2K128_FD=Format.from_shorthand("MXFP8[E5M2]{128,1}"),
    MXFP8_E5M2K64_LD=Format.from_shorthand("MXFP8[E5M2]{64,-1}"),
    MXFP8_E5M2K64_FD=Format.from_shorthand("MXFP8[E5M2]{64,1}"),
    MXFP8_E5M2K32_LD=Format.from_shorthand("MXFP8[E5M2]{32,-1}"),
    MXFP8_E5M2K32_FD=Format.from_shorthand("MXFP8[E5M2]{32,1}"),
    MXFP6_E2M3K128_LD=Format.from_shorthand("MXFP6[E2M3]{128,-1}"),
    MXFP6_E2M3K128_FD=Format.from_shorthand("MXFP6[E2M3]{128,1}"),
    MXFP6_E2M3K64_LD=Format.from_shorthand("MXFP6[E2M3]{64,-1}"),
    MXFP6_E2M3K64_FD=Format.from_shorthand("MXFP6[E2M3]{64,1}"),
    MXFP6_E2M3K32_LD=Format.from_shorthand("MXFP6[E2M3]{32,-1}"),
    MXFP6_E2M3K32_FD=Format.from_shorthand("MXFP6[E2M3]{32,1}"),
    MXFP6_E3M2K128_LD=Format.from_shorthand("MXFP6[E3M2]{128,-1}"),
    MXFP6_E3M2K128_FD=Format.from_shorthand("MXFP6[E3M2]{128,1}"),
    MXFP6_E3M2K64_LD=Format.from_shorthand("MXFP6[E3M2]{64,-1}"),
    MXFP6_E3M2K64_FD=Format.from_shorthand("MXFP6[E3M2]{64,1}"),
    MXFP6_E3M2K32_LD=Format.from_shorthand("MXFP6[E3M2]{32,-1}"),
    MXFP6_E3M2K32_FD=Format.from_shorthand("MXFP6[E3M2]{32,1}"),
    MXFP4_E2M1K128_LD=Format.from_shorthand("MXFP4[E2M1]{128,-1}"),
    MXFP4_E2M1K128_FD=Format.from_shorthand("MXFP4[E2M1]{128,1}"),
    MXFP4_E2M1K64_LD=Format.from_shorthand("MXFP4[E2M1]{64,-1}"),
    MXFP4_E2M1K64_FD=Format.from_shorthand("MXFP4[E2M1]{64,1}"),
    MXFP4_E2M1K32_LD=Format.from_shorthand("MXFP4[E2M1]{32,-1}"),
    MXFP4_E2M1K32_FD=Format.from_shorthand("MXFP4[E2M1]{32,1}"),
    MXINT8_K128_LD=Format.from_shorthand("MXINT8{128,-1}"),
    MXINT8_K128_FD=Format.from_shorthand("MXINT8{128,1}"),
    MXINT8_K64_LD=Format.from_shorthand("MXINT8{64,-1}"),
    MXINT8_K64_FD=Format.from_shorthand("MXINT8{64,1}"),
    MXINT8_K32_LD=Format.from_shorthand("MXINT8{32,-1}"),
    MXINT8_K32_FD=Format.from_shorthand("MXINT8{32,1}"),
    MXINT6_K128_LD=Format.from_shorthand("MXINT6{128,-1}"),
    MXINT6_K128_FD=Format.from_shorthand("MXINT6{128,1}"),
    MXINT6_K64_LD=Format.from_shorthand("MXINT6{64,-1}"),
    MXINT6_K64_FD=Format.from_shorthand("MXINT6{64,1}"),
    MXINT6_K32_LD=Format.from_shorthand("MXINT6{32,-1}"),
    MXINT6_K32_FD=Format.from_shorthand("MXINT6{32,1}"),
    MXINT4_K128_LD=Format.from_shorthand("MXINT4{128,-1}"),
    MXINT4_K128_FD=Format.from_shorthand("MXINT4{128,1}"),
    MXINT4_K64_LD=Format.from_shorthand("MXINT4{64,-1}"),
    MXINT4_K64_FD=Format.from_shorthand("MXINT4{64,1}"),
    MXINT4_K32_LD=Format.from_shorthand("MXINT4{32,-1}"),
    MXINT4_K32_FD=Format.from_shorthand("MXINT4{32,1}"),
)

# Sparseness aliases
sparseness = SimpleNamespace(
    BTK8_4_LD=Sparseness.from_shorthand("BTOPK{4:8,-1}(U)"),
    BTK8_4_FD=Sparseness.from_shorthand("BTOPK{4:8,1}(U)"),
    BTK8_2_LD=Sparseness.from_shorthand("BTOPK{2:8,-1}(U)"),
    BTK8_2_FD=Sparseness.from_shorthand("BTOPK{2:8,1}(U)"),
)

# Default approximation function aliases
default_approx = SimpleNamespace(
    RELU=ApproximationFunction.from_shorthand("NONE"),
    RELU6=ApproximationFunction.from_shorthand("NONE"),
    SILU=ApproximationFunction.from_shorthand("NONE"),
    SOFTMAX=ApproximationFunction.from_shorthand("NONE"),
    GELU=ApproximationFunction.from_shorthand("NONE"),
    TANH=ApproximationFunction.from_shorthand("NONE"),
    BATCHNORM2D=ApproximationFunction.from_shorthand("NONE"),
    LAYERNORM=ApproximationFunction.from_shorthand("NONE"),
    GROUPNORM=ApproximationFunction.from_shorthand("NONE"),
    RMSNORM=ApproximationFunction.from_shorthand("NONE"),
    LLAMAROTARYEMBEDDINGREFACTORED=ApproximationFunction.from_shorthand("NONE"),
    HFDIFFUSERSTIMESTEPS=ApproximationFunction.from_shorthand("NONE"),
)

# Automatic configuration rules
config_rules = SimpleNamespace(
    BASELINE=[],
    BASIC=[
        DmxConfigRule(
            module_types=(nn.Linear,),
            module_config=dict(
                input_format=format.BFP16_64,
                weight_format=format.BFP16_64,
                bias_format=format.BFP32_1,
                output_format=format.FLOAT16,
            ),
        ),
        DmxConfigRule(
            module_types=(
                nn.Conv1d,
                nn.Conv2d,
                nn.ConvTranspose2d,
            ),
            module_config=dict(
                input_format=format.BFP16_64,
                weight_format=format.BFP16_64,
                bias_format=format.BFP32_1,
                output_format=format.FLOAT16,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.ResAdd,),
            module_config=dict(
                input_format=format.FLOAT16,
                residual_format=format.FLOAT16,
                output_format=format.FLOAT16,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.ActActMatMul,),
            module_config=dict(
                input_format=format.BFP16_64,
                multiplier_format=format.BFP16_64,
                output_format=format.FLOAT16,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.Embedding,),
            module_config=dict(
                output_format=format.FLOAT16,
            ),
        ),
        DmxConfigRule(
            module_types=(
                nn.MaxPool2d,
                nn.AdaptiveAvgPool2d,
                nn.AvgPool2d,
            ),
            module_config=dict(
                input_format=format.FLOAT16,
                output_format=format.FLOAT16,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.ReLU,),
            module_config=dict(
                input_format=format.FLOAT16,
                output_format=format.FLOAT16,
                approximation_function=default_approx.RELU,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.ReLU6,),
            module_config=dict(
                input_format=format.FLOAT16,
                output_format=format.FLOAT16,
                approximation_function=default_approx.RELU6,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.GELU,),
            module_config=dict(
                input_format=format.FLOAT16,
                output_format=format.FLOAT16,
                approximation_function=default_approx.GELU,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.SiLU,),
            module_config=dict(
                input_format=format.FLOAT16,
                output_format=format.FLOAT16,
                approximation_function=default_approx.SILU,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.Tanh,),
            module_config=dict(
                input_format=format.FLOAT16,
                output_format=format.FLOAT16,
                approximation_function=default_approx.TANH,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.Softmax,),
            module_config=dict(
                input_format=format.FLOAT16,
                output_format=format.FLOAT16,
                approximation_function=default_approx.SOFTMAX,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.LayerNorm,),
            module_config=dict(
                input_format=format.FLOAT16,
                output_format=format.FLOAT16,
                approximation_function=default_approx.LAYERNORM,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.BatchNorm2d,),
            module_config=dict(
                input_format=format.FLOAT16,
                output_format=format.FLOAT16,
                approximation_function=default_approx.BATCHNORM2D,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.GroupNorm,),
            module_config=dict(
                input_format=format.FLOAT16,
                output_format=format.FLOAT16,
                approximation_function=default_approx.GROUPNORM,
            ),
        ),
    ],
)
