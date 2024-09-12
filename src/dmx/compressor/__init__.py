from types import SimpleNamespace
import torch
from functools import partialmethod

torch.nn.Module.load_state_dict = partialmethod(
    torch.nn.Module.load_state_dict, strict=False
)

from .numerical import Format
from .sparse import Sparseness
from .functional import ApproximationFunction
from .modeling import (
    nn,
    DmxConfigRule,
)

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
    BFP24_32=Format.from_shorthand("BFP[16|8]{32}(SN)"),
    BFP24_16=Format.from_shorthand("BFP[16|8]{16}(SN)"),
    BFP16_128=Format.from_shorthand("BFP[8|8]{128}(SN)"),
    BFP16_64=Format.from_shorthand("BFP[8|8]{64}(SN)"),
    BFP16_32=Format.from_shorthand("BFP[8|8]{32}(SN)"),
    BFP16_16=Format.from_shorthand("BFP[8|8]{16}(SN)"),
    BFP14_128=Format.from_shorthand("BFP[6|8]{128}(SN)"),
    BFP14_64=Format.from_shorthand("BFP[6|8]{64}(SN)"),
    BFP14_32=Format.from_shorthand("BFP[6|8]{32}(SN)"),
    BFP14_16=Format.from_shorthand("BFP[6|8]{16}(SN)"),
    BFP12_128=Format.from_shorthand("BFP[4|8]{128}(SN)"),
    BFP12_64=Format.from_shorthand("BFP[4|8]{64}(SN)"),
    BFP12_32=Format.from_shorthand("BFP[4|8]{32}(SN)"),
    BFP12_16=Format.from_shorthand("BFP[4|8]{16}(SN)"),
    BFP16A_128=Format.from_shorthand("BFP[8|8]{128}(_N)"),
    BFP16A_64=Format.from_shorthand("BFP[8|8]{64}(_N)"),
    BFP16A_32=Format.from_shorthand("BFP[8|8]{32}(_N)"),
    BFP16A_16=Format.from_shorthand("BFP[6|8]{16}(_N)"),
    BFP14A_128=Format.from_shorthand("BFP[6|8]{128}(_N)"),
    BFP14A_64=Format.from_shorthand("BFP[6|8]{64}(_N)"),
    BFP14A_32=Format.from_shorthand("BFP[6|8]{32}(_N)"),
    BFP14A_16=Format.from_shorthand("BFP[6|8]{16}(_N)"),
    BFP12A_128=Format.from_shorthand("BFP[4|8]{128}(_N)"),
    BFP12A_64=Format.from_shorthand("BFP[4|8]{64}(_N)"),
    BFP12A_32=Format.from_shorthand("BFP[4|8]{32}(_N)"),
    BFP12A_16=Format.from_shorthand("BFP[4|8]{16}(_N)"),
    SBFP12_16_4=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,4](FN)>{16}"),
    SBFP12_16_5=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,5](FN)>{16}"),
    SBFP12_16_6=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,6](FN)>{16}"),
    SBFP12_16_7=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,7](FN)>{16}"),
    SBFP12_16_8=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,8](FN)>{16}"),
    SBFP12_16_9=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,9](FN)>{16}"),
    SBFP12_16_10=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,10](FN)>{16}"),
    SBFP12_16_11=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,11](FN)>{16}"),
    SBFP12_16_12=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,12](FN)>{16}"),
    SBFP12_16_13=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,13](FN)>{16}"),
    SBFP12_16_14=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,14](FN)>{16}"),
    SBFP12_16_15=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,15](FN)>{16}"),
    SBFP12_16_16=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,16](FN)>{16}"),
    SBFP12_16_17=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,17](FN)>{16}"),
    SBFP12_16_18=Format.from_shorthand("SBFP<XP[4,0](CSN)><FP[0|4|4,18](FN)>{16}"),
    MXFP8_E4M3K128=Format.from_shorthand("MXFP8[E4M3]{128}"),
    MXFP8_E4M3K64=Format.from_shorthand("MXFP8[E4M3]{64}"),
    MXFP8_E4M3K32=Format.from_shorthand("MXFP8[E4M3]{32}"),
    MXFP8_E5M2K128=Format.from_shorthand("MXFP8[E5M2]{128}"),
    MXFP8_E5M2K64=Format.from_shorthand("MXFP8[E5M2]{64}"),
    MXFP8_E5M2K32=Format.from_shorthand("MXFP8[E5M2]{32}"),
    MXFP6_E2M3K128=Format.from_shorthand("MXFP6[E2M3]{128}"),
    MXFP6_E2M3K64=Format.from_shorthand("MXFP6[E2M3]{64}"),
    MXFP6_E2M3K32=Format.from_shorthand("MXFP6[E2M3]{32}"),
    MXFP6_E3M2K128=Format.from_shorthand("MXFP6[E3M2]{128}"),
    MXFP6_E3M2K64=Format.from_shorthand("MXFP6[E3M2]{64}"),
    MXFP6_E3M2K32=Format.from_shorthand("MXFP6[E3M2]{32}"),
    MXFP4_E2M1K128=Format.from_shorthand("MXFP4[E2M1]{128}"),
    MXFP4_E2M1K64=Format.from_shorthand("MXFP4[E2M1]{64}"),
    MXFP4_E2M1K32=Format.from_shorthand("MXFP4[E2M1]{32}"),
    MXINT8_K128=Format.from_shorthand("MXINT8{128}"),
    MXINT8_K64=Format.from_shorthand("MXINT8{64}"),
    MXINT8_K32=Format.from_shorthand("MXINT8{32}"),
    MXINT6_K128=Format.from_shorthand("MXINT6{128}"),
    MXINT6_K64=Format.from_shorthand("MXINT6{64}"),
    MXINT6_K32=Format.from_shorthand("MXINT6{32}"),
    MXINT4_K128=Format.from_shorthand("MXINT4{128}"),
    MXINT4_K64=Format.from_shorthand("MXINT4{64}"),
    MXINT4_K32=Format.from_shorthand("MXINT4{32}"),
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
                input_formats=[format.BFP16_64],
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
                input_formats=[format.BFP16_64],
                weight_format=format.BFP16_64,
                bias_format=format.BFP32_1,
                output_format=format.FLOAT16,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.ResAdd,),
            module_config=dict(
                input_formats=[format.FLOAT16, format.FLOAT16],
                output_format=format.FLOAT16,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.ActActMatMul,),
            module_config=dict(
                input_formats=[format.BFP16_64, format.BFP16_64],
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
                input_formats=[format.FLOAT16],
                output_format=format.FLOAT16,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.ReLU,),
            module_config=dict(
                input_formats=[format.FLOAT16],
                output_format=format.FLOAT16,
                approximation_function=default_approx.RELU,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.ReLU6,),
            module_config=dict(
                input_formats=[format.FLOAT16],
                output_format=format.FLOAT16,
                approximation_function=default_approx.RELU6,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.GELUBase,),
            module_config=dict(
                input_formats=[format.FLOAT16],
                output_format=format.FLOAT16,
                approximation_function=default_approx.GELU,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.SiLU,),
            module_config=dict(
                input_formats=[format.FLOAT16],
                output_format=format.FLOAT16,
                approximation_function=default_approx.SILU,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.Tanh,),
            module_config=dict(
                input_formats=[format.FLOAT16],
                output_format=format.FLOAT16,
                approximation_function=default_approx.TANH,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.Softmax,),
            module_config=dict(
                input_formats=[format.FLOAT16],
                output_format=format.FLOAT16,
                approximation_function=default_approx.SOFTMAX,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.LayerNorm,),
            module_config=dict(
                input_formats=[format.FLOAT16],
                output_format=format.FLOAT16,
                approximation_function=default_approx.LAYERNORM,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.BatchNorm2d,),
            module_config=dict(
                input_formats=[format.FLOAT16],
                output_format=format.FLOAT16,
                approximation_function=default_approx.BATCHNORM2D,
            ),
        ),
        DmxConfigRule(
            module_types=(nn.GroupNorm,),
            module_config=dict(
                input_formats=[format.FLOAT16],
                output_format=format.FLOAT16,
                approximation_function=default_approx.GROUPNORM,
            ),
        ),
    ],
)
