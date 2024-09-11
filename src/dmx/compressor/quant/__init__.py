try:
    from .quant_function import *
except ImportError as error:
    print("Error importing Block Quantize CUDA kernels")

__all__ = [
    "fixed_point_quantize",
    "block_quantize",
    "float_quantize",
]
