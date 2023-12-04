from types import SimpleNamespace
from ..numerical import CastTo, Format
from ..sparse import Sparsify, Sparseness
from ..functional import Approximate, ApproximationFunction
from .model import Model, aware, DmxConfig, DmxTransformation
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
    BFP32_1=Format.from_shorthand("BFP[24|8]{1,-1}(SN)"),
    BFP24_64_LD=Format.from_shorthand("BFP[16|8]{64,-1}(SN)"),
    BFP24_64_FD=Format.from_shorthand("BFP[16|8]{64,1}(SN)"),
    BFP24_32_LD=Format.from_shorthand("BFP[16|8]{32,-1}(SN)"),
    BFP24_32_FD=Format.from_shorthand("BFP[16|8]{32,1}(SN)"),
    BFP24_16_LD=Format.from_shorthand("BFP[16|8]{16,-1}(SN)"),
    BFP24_16_FD=Format.from_shorthand("BFP[16|8]{16,1}(SN)"),
    BFP16_128_LD=Format.from_shorthand("BFP[8|8]{128,-1}(SN)"),
    BFP16_128_FD=Format.from_shorthand("BFP[8|8]{128,1}(SN)"),
    BFP16_64_LD=Format.from_shorthand("BFP[8|8]{64,-1}(SN)"),
    BFP16_64_FD=Format.from_shorthand("BFP[8|8]{64,1}(SN)"),
    BFP16_32_LD=Format.from_shorthand("BFP[8|8]{32,-1}(SN)"),
    BFP16_32_FD=Format.from_shorthand("BFP[8|8]{32,1}(SN)"),
    BFP16_16_LD=Format.from_shorthand("BFP[8|8]{16,-1}(SN)"),
    BFP16_16_FD=Format.from_shorthand("BFP[8|8]{16,1}(SN)"),
    BFP14_128_LD=Format.from_shorthand("BFP[6|8]{128,-1}(SN)"),
    BFP14_128_FD=Format.from_shorthand("BFP[6|8]{128,1}(SN)"),
    BFP14_64_LD=Format.from_shorthand("BFP[6|8]{64,-1}(SN)"),
    BFP14_64_FD=Format.from_shorthand("BFP[6|8]{64,1}(SN)"),
    BFP14_32_LD=Format.from_shorthand("BFP[6|8]{32,-1}(SN)"),
    BFP14_32_FD=Format.from_shorthand("BFP[6|8]{32,1}(SN)"),
    BFP14_16_LD=Format.from_shorthand("BFP[6|8]{16,-1}(SN)"),
    BFP14_16_FD=Format.from_shorthand("BFP[6|8]{16,1}(SN)"),
    BFP12_128_LD=Format.from_shorthand("BFP[4|8]{128,-1}(SN)"),
    BFP12_128_FD=Format.from_shorthand("BFP[4|8]{128,1}(SN)"),
    BFP12_64_LD=Format.from_shorthand("BFP[4|8]{64,-1}(SN)"),
    BFP12_64_FD=Format.from_shorthand("BFP[4|8]{64,1}(SN)"),
    BFP12_32_LD=Format.from_shorthand("BFP[4|8]{32,-1}(SN)"),
    BFP12_32_FD=Format.from_shorthand("BFP[4|8]{32,1}(SN)"),
    BFP12_16_LD=Format.from_shorthand("BFP[4|8]{16,-1}(SN)"),
    BFP12_16_FD=Format.from_shorthand("BFP[4|8]{16,1}(SN)"),
    BFP16A_128_LD=Format.from_shorthand("BFP[8|8]{128,-1}(_N)"),
    BFP16A_128_FD=Format.from_shorthand("BFP[8|8]{128,1}(_N)"),
    BFP16A_64_LD=Format.from_shorthand("BFP[8|8]{64,-1}(_N)"),
    BFP16A_64_FD=Format.from_shorthand("BFP[8|8]{64,1}(_N)"),
    BFP16A_32_LD=Format.from_shorthand("BFP[8|8]{32,-1}(_N)"),
    BFP16A_32_FD=Format.from_shorthand("BFP[8|8]{32,1}(_N)"),
    BFP16A_16_LD=Format.from_shorthand("BFP[8|8]{16,-1}(_N)"),
    BFP16A_16_FD=Format.from_shorthand("BFP[6|8]{16,1}(_N)"),
    BFP14A_128_LD=Format.from_shorthand("BFP[6|8]{128,-1}(_N)"),
    BFP14A_128_FD=Format.from_shorthand("BFP[6|8]{128,1}(_N)"),
    BFP14A_64_LD=Format.from_shorthand("BFP[6|8]{64,-1}(_N)"),
    BFP14A_64_FD=Format.from_shorthand("BFP[6|8]{64,1}(_N)"),
    BFP14A_32_LD=Format.from_shorthand("BFP[6|8]{32,-1}(_N)"),
    BFP14A_32_FD=Format.from_shorthand("BFP[6|8]{32,1}(_N)"),
    BFP14A_16_LD=Format.from_shorthand("BFP[6|8]{16,-1}(_N)"),
    BFP14A_16_FD=Format.from_shorthand("BFP[6|8]{16,1}(_N)"),
    BFP12A_128_LD=Format.from_shorthand("BFP[4|8]{128,-1}(_N)"),
    BFP12A_128_FD=Format.from_shorthand("BFP[4|8]{128,1}(_N)"),
    BFP12A_64_LD=Format.from_shorthand("BFP[4|8]{64,-1}(_N)"),
    BFP12A_64_FD=Format.from_shorthand("BFP[4|8]{64,1}(_N)"),
    BFP12A_32_LD=Format.from_shorthand("BFP[4|8]{32,-1}(_N)"),
    BFP12A_32_FD=Format.from_shorthand("BFP[4|8]{32,1}(_N)"),
    BFP12A_16_LD=Format.from_shorthand("BFP[4|8]{16,-1}(_N)"),
    BFP12A_16_FD=Format.from_shorthand("BFP[4|8]{16,1}(_N)"),
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
    SOFTMAX=ApproximationFunction.from_shorthand("SOFTMAX(vsimd)"),
    GELU=ApproximationFunction.from_shorthand("GELU(vsimd)"),
    LAYERNORM=ApproximationFunction.from_shorthand("LAYERNORM(vsimd)"),
    T5LAYERNORM=ApproximationFunction.from_shorthand("NONE"),
    LLAMALAYERNORM=ApproximationFunction.from_shorthand("NONE"),
    LLAMAROTARYEMBEDDINGREFACTORED=ApproximationFunction.from_shorthand("NONE"),
    HFDIFFUSERSTIMESTEPS=ApproximationFunction.from_shorthand("HFDIFFUSERSTIMESTEPS(vsimd)"),
)