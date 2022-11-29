from types import SimpleNamespace
import torch
from ..numerical import CastTo, Format
from ..sparse import Sparsify, Sparseness
from ..approximate import Approximate, ApproximationFunction
from .transform import Model, aware, CorsairConfig, CorsairTransformation
from . import nn

# Numerical format aliases
format = SimpleNamespace(
    FLOAT32=Format.from_shorthand("FP[1|8|23,127](FN)"),
    FLOAT16=Format.from_shorthand("FP[1|5|10,15](FN)"),
    BFLOAT16=Format.from_shorthand("FP[1|8|7,127](FN)"),
    AFLOAT8=Format.from_shorthand("FP[1|4|3,7](_N)"),
    BFLOAT8=Format.from_shorthand("FP[1|5|2,15](_N)"),
    INT8=Format.from_shorthand("XP[8,0](CSN)"),
    INT4=Format.from_shorthand("XP[4,0](CSN)"),
    BFP16_64_LD=Format.from_shorthand("BFP[8|8]{64,-1}(SN)"),
    BFP16_64_FD=Format.from_shorthand("BFP[8|8]{64,1}(SN)"),
    BFP16_128_LD=Format.from_shorthand("BFP[8|8]{128,-1}(SN)"),
    BFP16_128_FD=Format.from_shorthand("BFP[8|8]{128,1}(SN)"),
    BFP12_128_LD=Format.from_shorthand("BFP[4|8]{128,-1}(SN)"),
    BFP12_128_FD=Format.from_shorthand("BFP[4|8]{128,1}(SN)"),
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
    SOFTMAX=ApproximationFunction.from_shorthand("SOFTMAX(base2,float16)"),
    GELU=ApproximationFunction.from_shorthand("GELU(poly2,float16)"),
    LAYERNORM=ApproximationFunction.from_shorthand("LAYERNORM(quake3,float16)"),
)


class Chiplet:
    def __init__(self):
        pass


class Quad:
    def __init__(self):
        pass


class Slice:
    def __init__(self):
        pass
