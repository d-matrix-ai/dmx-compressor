import warnings
from typing import Union, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.quantization.fake_quantize import FakeQuantize
import transformers
from .format import (
    Format,
    Same,
    FixedPoint,
    FloatingPoint,
    BlockFloatingPoint,
    ScaledBlockFloatingPoint,
)
from .smoothquant import ActivationWeightSmoothQuant
from .observer import DummyObserver


class CastToFormat(Function):
    r"""
    A simple STE backward function for numerical cast
    """

    @staticmethod
    def forward(ctx, x, fmt):
        ctx.set_materialize_grads(False)
        return fmt.cast(x)

    @staticmethod
    def backward(ctx, g):
        return g, None

    @staticmethod
    def symbolic(
        g: torch._C.Graph, input: torch._C.Value, fmt: torch._C.Value
    ) -> torch._C.Value:
        if isinstance(fmt, Same):
            return g.op("Identity", input)
        elif isinstance(fmt, BlockFloatingPoint):
            # TODO with dtype for torch > 1.11
            return g.op(
                "com.microsoft::DequantizeBFP",
                *g.op(
                    "com.microsoft::QuantizeBFP",
                    input,
                    bfp_type_i=torch.onnx.symbolic_helper._parse_arg(fmt.bfp_id, "i"),
                    outputs=3,
                ),
                bfp_type_i=torch.onnx.symbolic_helper._parse_arg(fmt.bfp_id, "i"),
                dtype_i=1,
                outputs=1,
            )
        else:
            return None


class CastTo(FakeQuantize):
    r"""
    Simulated numerical cast to a target format
    subclassing torch.quantization.fake_quantize.FakeQuantize
    TODO: special state_dict handling to include and exclude flags
    """

    def __init__(self, format="SAME", observer=DummyObserver, **fake_quantize_kwargs):
        self.set_format(format)
        super().__init__(observer=observer, dtype=self.format, **fake_quantize_kwargs)
        self.physical_dtype = None
        self.enable_fake_quant()
        self.disable_observer()

    def set_format(self, format: Union[str, torch.dtype, Format]):
        if isinstance(format, str):
            format = Format.from_shorthand(format)
        self.format = format
        if hasattr(self, "dtype"):
            self.dtype = format
            self.activation_post_process.dtype = format
            if hasattr(format, "block_dim"):
                self.activation_post_process.ch_axis = self.ch_axis = format.block_dim

    def _observer_step(self, x):
        r"""
        Helper method for stepping observer in forward(),
        taken from torch.quantization.fake_quantize.FakeQuantize source
        """
        self.activation_post_process(x.detach())
        _scale, _zero_point = self.calculate_qparams()
        _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(
            self.zero_point.device
        )
        if self.scale.shape != _scale.shape:
            self.scale.resize_(_scale.shape)
            self.zero_point.resize_(_zero_point.shape)
        self.scale.copy_(_scale)
        self.zero_point.copy_(_zero_point)

    def _get_affine_params(self, x):
        if self.is_per_channel:
            extended_shape = [1] * x.dim()
            channel_dim = x.shape[self.ch_axis]
            extended_shape[self.ch_axis] = channel_dim
            sc = self.scale[:channel_dim].view(extended_shape)
            zp = self.zero_point[:channel_dim].view(extended_shape)
        else:
            sc, zp = self.scale, self.zero_point
        return sc.to(x.device), zp.to(x.device)

    def forward(self, x):
        self.physical_dtype = x.dtype
        if self.observer_enabled[0] == 1 and x is not None:
            self._observer_step(x)
        if self.fake_quant_enabled[0] == 1:
            if isinstance(self.format, Format):  # d-Matrix custom format
                if isinstance(self.format, FixedPoint):
                    sc, zp = self._get_affine_params(x)
                    x = x / sc + zp
                x = CastToFormat.apply(x, self.format)
                if isinstance(self.format, FixedPoint):
                    x = (x - zp) * sc
            else:  # torch.dtype
                x = super().forward(x)
        return x

    def get_precision(self) -> Optional[int]:
        if isinstance(self.format, (Same, torch.dtype)):
            if self.physical_dtype is not None:
                return torch.finfo(self.physical_dtype).bits
            else:
                raise RuntimeError(
                    "physical_dtype has not been inferred, pass some data through first"
                )
        else:
            return self.format.bit_precision

    def extra_repr(self):
        return f"format = dtype = {repr(self.format)}, qscheme = {self.qscheme}, ch_axis = {self.ch_axis} \nfake_quant_enabled = {bool(self.fake_quant_enabled)}, observer_enabled = {bool(self.observer_enabled)}, scale = {self.scale.cpu().numpy()}, zero_point = {self.zero_point.cpu().numpy()}"

from torch_mlir_e2e_test.annotations import export
class Quantize(torch.nn.quantized.Quantize):
    r"""Drop-in replacement of torch.nn.quantized.Quantize
    that supports both torch.dtype and numerical.Format

    Args:
     `scale`: scale of the output Quantized Tensor
     `zero_point`: zero_point of output Quantized Tensor
     `dtype`: data type of output Quantized Tensor, either torch.dtype or Format

    Attributes:
      `scale`, `zero_point`, `dtype`
    """

    def __init__(self, scale, zero_point, dtype: Union[torch.dtype, Format]):
        super().__init__(scale, zero_point, dtype)

    @export
    def forward(self, x):
        return torch.ops.dmx.quantize(x)

class DeQuantize(torch.nn.quantized.DeQuantize):
    r"""Drop-in replacement of torch.nn.quantized.DeQuantize
    that supports both torch.dtype and numerical.Format

    Args:
     `scale`: scale of the output Quantized Tensor
     `zero_point`: zero_point of output Quantized Tensor
     `dtype`: data type of output Quantized Tensor, either torch.dtype or Format

    Attributes:
      `scale`, `zero_point`, `dtype`
    """

    def __init__(self, scale=None, zero_point=None, dtype=None):
        super().__init__()

    def forward(self, x):
        return torch.ops.dmx.dequantize(x)

class NumericalCastMixin:
    r"""
    Mixin for modules with boundary casting
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.align_boundary_dtype = True
        self.align_boundary_device = True
        self.infer_ch_axis()
        self.init_casts()
        self.init_smoothquant()

    def infer_ch_axis(self):
        if isinstance(
            self,
            (nn.Linear,),
        ):
            self.ch_axis = -1
            self.w_ch_axis = -1
        elif isinstance(
            self,
            (transformers.pytorch_utils.Conv1D,),
        ):
            self.ch_axis = -1
            self.w_ch_axis = 0
        elif isinstance(
            self,
            (nn.modules.conv._ConvNd,),
        ):
            self.ch_axis = 1
            self.w_ch_axis = 1
        else:
            self.ch_axis = self.w_ch_axis = None

    def init_casts(self) -> None:
        # dynamic i/o casts
        self.input_cast = CastTo()
        self.output_cast = CastTo()
        # dynamic intermediate casts
        if isinstance(
            self,
            (
                nn.Linear,
                nn.Bilinear,
                nn.EmbeddingBag,
                nn.modules.conv._ConvNd,
            ),
        ):
            self.accum_cast = CastTo()
        else:
            self.accum_cast = None
        # static parameter casts
        pnames = [n for n, _ in self.named_parameters()]
        self.weight_cast = (
            CastTo(ch_axis=self.w_ch_axis) if "weight" in pnames else None
        )
        self.bias_cast = CastTo() if "bias" in pnames else None
        self.residual_cast = None
        self.multiplier_cast = None

    def init_smoothquant(
        self,
        migration_strength: float = 0.5,
        scale_format: Union[str, Format] = "SAME",
        dynamic: bool = False,
    ) -> None:
        self.smoothquant = (
            ActivationWeightSmoothQuant(
                self.ch_axis, self.w_ch_axis, migration_strength, scale_format, dynamic
            )
            if self.ch_axis is not None and self.w_ch_axis is not None
            else None
        )

    @staticmethod
    def _check_format_dim_consistency(format: Format, ch_axis: int) -> bool:
        return not format.blocked or format.block_dim == ch_axis

    def check_weight_format_dim_consistency(self) -> bool:
        _good = self.weight_cast is None or self._check_format_dim_consistency(
            self.weight_cast.format, self.w_ch_axis
        )
        if not _good:
            warnings.warn(
                f"layer's weight channel axis {self.w_ch_axis} might be inconsistent with format {self.weight_format}",
                RuntimeWarning,
            )
        return _good

    def check_input_format_dim_consistency(self) -> bool:
        _good = self.input_cast is None or self._check_format_dim_consistency(
            self.input_cast.format, self.ch_axis
        )
        if not _good:
            warnings.warn(
                f"layer's input channel axis {self.ch_axis} might be inconsistent with format {self.input_format}",
                RuntimeWarning,
            )
        return _good

    def check_residual_format_dim_consistency(self) -> bool:
        _good = self.residual_cast is None or self._check_format_dim_consistency(
            self.residual_cast.format, self.ch_axis
        )
        if not _good:
            warnings.warn(
                f"layer's residual channel axis {self.ch_axis} might be inconsistent with format {self.residual_format}",
                RuntimeWarning,
            )
        return _good

    def check_format_dim_consistency(self) -> bool:
        return (
            self.check_input_format_dim_consistency()
            and self.check_residual_format_dim_consistency()
            and self.check_weight_format_dim_consistency()
        )

    @property
    def input_format(self):
        return self.input_cast.format

    @property
    def input_precision(self):
        return self.input_cast.get_precision()

    @property
    def weight_precision(self):
        return self.weight_cast.get_precision()

    @property
    def output_format(self):
        return self.output_cast.format

    @property
    def accum_format(self):
        return (self.accum_cast.format) if self.accum_cast is not None else None

    @property
    def weight_format(self):
        return (self.weight_cast.format) if self.weight_cast is not None else None

    @property
    def residual_format(self):
        return (self.residual_cast.format) if self.residual_cast is not None else None

    @property
    def multiplier_format(self):
        return (
            (self.multiplier_cast.format) if self.multiplier_cast is not None else None
        )

    @property
    def bias_format(self):
        return (self.bias_cast.format) if self.bias_cast is not None else None
