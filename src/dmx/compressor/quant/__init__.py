try:
    from .quant_function import *
except ImportError as error:
    print(error.__class__.__name__ + ": " + error.message)
except Exception as exception:
    print(exception, False)
    print(exception.__class__.__name__ + ": " + exception.message)

__all__ = [
    "fixed_point_quantize",
    "block_quantize",
    "float_quantize",
]
