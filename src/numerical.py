import torch
import torch.nn as nn
import qtorch


__ALL__ = ["IMCOp", "SIMDOp"]

IMC_INPUT_FORMAT = qtorch.BlockFloatingPoint(wl=8, dim=1)
IMC_PARAM_FORMAT = qtorch.BlockFloatingPoint(wl=8, dim=1)
IMC_OUTPUT_FORMAT = qtorch.FloatingPoint(exp=8, man=23)

SIMD_INPUT_FORMAT = qtorch.FixedPoint(wl=24, fl=8)
SIMD_INPUT_FORMAT = qtorch.FixedPoint(wl=24, fl=8)
SIMD_OUTPUT_FORMAT = qtorch.FixedPoint(wl=24, fl=8)


class SIMDOp:
    pass


class IMCOp:
    pass