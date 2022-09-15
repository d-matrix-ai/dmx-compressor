from types import SimpleNamespace
import torch
from ..numerical import CastTo, Format
from ..sparse import Sparsify, Sparseness
from ..approximate import Approximate, ApproximationFunction
from ..dmir import discard_values
from .transform import Model, aware, CorsairConfig, CorsairTransformation
from . import nn

# Numerical format aliases
format = SimpleNamespace(
    FLOAT32=Format.from_shorthand("FP[1|8|23]{127}(N)"),
    FLOAT16=Format.from_shorthand("FP[1|5|10]{15}(N)"),
    BFLOAT16=Format.from_shorthand("FP[1|8|7]{127}(N)"),
    AFLOAT8=Format.from_shorthand("FP[1|4|3]{7}(N)"),
    BFLOAT8=Format.from_shorthand("FP[1|5|2]{15}(N)"),
    INT8=Format.from_shorthand("XP[8,0](CSN)"),
    INT4=Format.from_shorthand("XP[4,0](CSN)"),
    BFP16_64_LD=Format.from_shorthand("BFP[8|8]{64,-1}(N)"),
    BFP16_64_FD=Format.from_shorthand("BFP[8|8]{64,1}(N)"),
    BFP12_128_LD=Format.from_shorthand("BFP[4|8]{128,-1}(N)"),
    BFP12_128_FD=Format.from_shorthand("BFP[4|8]{128,1}(N)"),
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
    GELU="GELU(poly2,float16)",
    LAYERNORM="LAYERNORM(quake3,float16)",
)


counterpart = {
    torch.nn.Linear: nn.Linear,
    torch.nn.Conv2d: nn.Conv2d,
    torch.nn.AdaptiveAvgPool2d: nn.AdaptiveAvgPool2d,
    torch.nn.MaxPool2d: nn.MaxPool2d,
    torch.nn.BatchNorm2d: nn.BatchNorm2d,
    torch.nn.LayerNorm: nn.LayerNorm,
    torch.nn.Dropout: nn.Dropout,
    torch.nn.Softmax: nn.Softmax,
    torch.nn.ReLU: nn.ReLU,
    torch.nn.ReLU6: nn.ReLU6,
    torch.nn.Tanh: nn.Tanh,
}

