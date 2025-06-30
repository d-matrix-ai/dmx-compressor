import warnings
from typing import Union, Optional, Dict
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.quantization.fake_quantize import FakeQuantize
import transformers
from .format import (
    Format,
    Same,
    FixedPoint,
    BlockFloatingPoint,
)
from .smoothquant import ActivationWeightSmoothQuant
from .observer import ObserverBase, DummyObserver, HistogramObserver
import math
from collections import OrderedDict


class CastToFormat(Function):
    r"""
    A simple STE backward function for numerical cast
    """

    @staticmethod
    def forward(ctx, x, fmt, block_dim):
        ctx.set_materialize_grads(False)
        return fmt.cast(x, block_dim)

    @staticmethod
    def backward(ctx, g):
        return g, None, None

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


class CastToDict(torch.nn.ModuleDict):
    def forward(
        self,
        x,
        *args,
        output=False,
        **kwargs,
    ):
        keys = list(self.keys())
        if output:
            if isinstance(x, (tuple, list)):
                return type(x)(self[keys[i]](a) for i, a in enumerate(x))
            return self[keys[0]](x)

        i = 1
        new_args = []
        new_kwargs = {}
        for a in args:
            if isinstance(a, torch.Tensor):
                new_args.append(self[keys[i]](a))
                i += 1
            else:
                new_args.append(a)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                new_kwargs[k] = self[k + "_cast"](v)
            else:
                new_kwargs[k] = v
        return self[keys[0]](x), new_args, new_kwargs

    def pack_to_dict(self, param):
        keys = list(self.keys())
        if isinstance(param, (tuple, list)):
            param = {
                keys[i]: p if p is not None else "SAME" for i, p in enumerate(param)
            }
        elif not isinstance(param, dict):
            raise ValueError("format needs to be a dict, tuple or list!")
        if len(param) != len(self):
            warnings.warn(
                f"length of format to set is not equal to length of input_casts, some CastTos might not be set properly!\nlen({param}!={len(self)})"
            )
        return param

    def set_format(self, format: Union[Dict, tuple, list]):
        format = self.pack_to_dict(format)
        for k, f in format.items():
            if isinstance(f, str):
                f = Format.from_shorthand(f)
            if k not in self.keys():
                raise RuntimeError(f"No CastTo with key {k}!")
            self[k].format = f
            if hasattr(self[k], "dtype"):
                self[k].dtype = f
                self[k].activation_post_process.dtype = f

    def disable_fake_quant(self):
        for k in self.keys():
            self[k].disable_fake_quant()

    def enable_fake_quant(self):
        for k in self.keys():
            self[k].enable_fake_quant()

    def enable_observer(self):
        for k in self.keys():
            self[k].enable_observer()

    def disable_observer(self):
        for k in self.keys():
            self[k].disable_observer()


class CastTo(FakeQuantize):
    r"""
    Simulated numerical cast to a target format
    subclassing torch.quantization.fake_quantize.FakeQuantize
    TODO: special state_dict handling to include and exclude flags
    """

    def __init__(
        self,
        format="SAME",
        observer=DummyObserver,
        group_size=None,
        block_dim=-1,
        **fake_quantize_kwargs,
    ):
        self.set_format(format)
        super().__init__(observer=observer, dtype=self.format, **fake_quantize_kwargs)
        if group_size:
            assert torch.ao.quantization.utils.is_per_tensor(
                self.qscheme
            ), "group_size must be used with per tensor quantization scheme"
        self.group_size = group_size if group_size else None
        self.physical_dtype = None
        self.block_dim = block_dim
        self.enable_fake_quant()
        self.disable_observer()

    def set_format(self, format: Union[str, torch.dtype, Format]):
        if isinstance(format, str):
            format = Format.from_shorthand(format)
        self.format = format
        if hasattr(self, "dtype"):
            self.dtype = format
            self.activation_post_process.dtype = format

    def _observer_step(self, x):
        r"""
        Helper method for stepping observer in forward(),
        taken from torch.quantization.fake_quantize.FakeQuantize source
        """
        self.activation_post_process.to(x.device)
        if self.group_size:
            if not hasattr(self, "activation_post_processes"):
                group_num = math.ceil(x.shape[self.ch_axis] * 1.0 / self.group_size)
                self.activation_post_processes = [
                    self.activation_post_process.__class__(
                        dtype=self.format,
                        qscheme=self.qscheme,
                        ch_axis=self.ch_axis,
                    ).to(x.device)
                    for i in range(group_num)
                ]
            xs = torch.split(
                x,
                self.group_size,
                dim=self.ch_axis,
            )
            scale, zero_point = [], []
            mins, maxs = [], []
            for i in range(len(xs)):
                self.activation_post_processes[i](xs[i])
                s, zp = self.activation_post_processes[i].calculate_qparams()
                scale.append(s)
                zero_point.append(zp)
                mins.append(self.activation_post_processes[i].min_val)
                maxs.append(self.activation_post_processes[i].max_val)
            _scale = torch.tensor(scale)
            _zero_point = torch.tensor(zero_point)
            self.activation_post_process.min_val = torch.tensor(mins)
            self.activation_post_process.max_val = torch.tensor(maxs)

        else:
            self.activation_post_process(x.detach().float())
            _scale, _zero_point = self.calculate_qparams()

        _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(
            self.zero_point.device
        )
        if self.scale.shape != _scale.shape:
            self.scale = torch.zeros_like(_scale)
            self.zero_point = torch.zeros_like(_zero_point)
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
        if (
            self.observer_enabled[0] == 1
            and x is not None
            and not isinstance(self.format, Same)
        ):
            self._observer_step(x)
        if self.fake_quant_enabled[0] == 1:
            if isinstance(self.format, Format):  # d-Matrix custom format
                if isinstance(self.format, FixedPoint):
                    sc, zp = self._get_affine_params(x)
                    if self.group_size:
                        # duplicate each element in sc and zp for group_size number of times
                        sc = torch.repeat_interleave(sc, self.group_size)[
                            : x.shape[self.ch_axis]
                        ]
                        zp = torch.repeat_interleave(zp, self.group_size)[
                            : x.shape[self.ch_axis]
                        ]
                        sc_shape = [1] * len(x.shape)
                        sc_shape[self.ch_axis] = x.shape[self.ch_axis]
                        sc = sc.view(sc_shape)
                        zp = zp.view(sc_shape)
                    x = x / sc + zp
                x = CastToFormat.apply(x, self.format, self.block_dim)
                if isinstance(self.format, FixedPoint):
                    x = (x - zp) * sc
            else:  # torch.dtype
                x = super().forward(x)
        return x.to(self.physical_dtype)

    def enable_calibration(
        self,
        state: bool = True,
        observer_cls: ObserverBase = HistogramObserver,
        qscheme_to_overload: Optional[torch.qscheme] = None,
        group_size: int = None,
        ch_axis: int = None,
    ) -> None:
        if state: 
            if ch_axis is not None:
                self.ch_axis = (
                    self.activation_post_process.ch_axis
                ) = ch_axis
            if qscheme_to_overload is not None:
                self.qscheme = qscheme_to_overload
                self.is_per_channel = (
                    torch.ao.quantization.utils.is_per_channel(qscheme_to_overload)
                )
            self.group_size = group_size if group_size else None
            if self.group_size:
                assert torch.ao.quantization.utils.is_per_tensor(
                    qscheme_to_overload
                ), "group quantization is to be used with per tensor quantization"
            self.activation_post_process = observer_cls(
                dtype=self.format,
                qscheme=self.qscheme,
                ch_axis=self.ch_axis,
            )
            self.disable_fake_quant()
            self.enable_observer()
        else:
            self.enable_fake_quant()
            self.disable_observer()


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
        if self.format.blocked:
            return f"format = dtype = {repr(self.format)}, block_dim = {self.block_dim} \nfake_quant_enabled = {bool(self.fake_quant_enabled)}"
        else:
            return f"format = dtype = {repr(self.format)}, qscheme = {self.qscheme}, ch_axis = {self.ch_axis} \nfake_quant_enabled = {bool(self.fake_quant_enabled)}, observer_enabled = {bool(self.observer_enabled)}, scale = {self.scale.cpu().numpy()}, zero_point = {self.zero_point.cpu().numpy()}, group_size = {self.group_size}"


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
        self.infer_ch_axis()
        self.init_casts()
        self.init_smoothquant()

    def infer_ch_axis(self):
        if isinstance(
            self,
            (nn.Linear,),
        ):
            self.ch_axis = -1
            self.win_ch_axis = -1
            self.wout_ch_axis = 0
        elif isinstance(
            self,
            (transformers.pytorch_utils.Conv1D,),
        ):
            self.ch_axis = -1
            self.win_ch_axis = 0
            self.wout_ch_axis = -1
        elif isinstance(
            self,
            (nn.modules.conv._ConvNd,),
        ):
            self.ch_axis = 1
            self.win_ch_axis = 1
            self.wout_ch_axis = 0
        else:
            self.ch_axis = self.win_ch_axis = self.wout_ch_axis = None

    def init_casts(self) -> None:
        # dynamic i/o casts
        self.input_casts = CastToDict(
            OrderedDict({"input_cast": CastTo(ch_axis=self.ch_axis)})
        )
        self.output_casts = CastToDict(OrderedDict({"output_cast": CastTo()}))
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
        self.weight_storage_cast = (
            CastTo(ch_axis=self.wout_ch_axis) if "weight" in pnames else None
        )
        self.weight_cast = (
            CastTo(ch_axis=self.wout_ch_axis) if "weight" in pnames else None
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
                self.ch_axis,
                self.win_ch_axis,
                migration_strength,
                scale_format,
                dynamic,
            )
            if self.ch_axis is not None and self.win_ch_axis is not None
            else None
        )

    @staticmethod
    def _check_format_dim_consistency(format: Format, ch_axis: int) -> bool:
        return not format.blocked or format.block_dim == ch_axis

    def check_weight_format_dim_consistency(self) -> bool:
        _good = self.weight_cast is None or self._check_format_dim_consistency(
            self.weight_cast.format, self.win_ch_axis
        )
        if not _good:
            warnings.warn(
                f"layer's weight input channel axis {self.win_ch_axis} might be inconsistent with format {self.weight_format}",
                RuntimeWarning,
            )
        return _good

    def check_input_format_dim_consistency(self) -> bool:
        _good = (
            self.input_casts is None
            or self.input_casts.input_cast is None
            or self._check_format_dim_consistency(
                self.input_casts.input_cast.format, self.ch_axis
            )
        )
        if not _good:
            warnings.warn(
                f"Input format dim consistency check failed! Layer's input channel axis {self.ch_axis} might be inconsistent with format {self.input_formats}",
                RuntimeWarning,
            )
        return _good

    def check_residual_format_dim_consistency(self) -> bool:
        _good = (
            self.input_casts is None
            or self.input_casts.residual_cast is None
            or self._check_format_dim_consistency(
                self.input_casts.residual_cast.format, self.ch_axis
            )
        )
        if not _good:
            warnings.warn(
                f"Residual format dim consistency check failed! Layer's input channel axis {self.ch_axis} might be inconsistent with format {self.input_formats}",
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
    def input_formats(self):
        return {k: cast.format for k, cast in self.input_casts.items()}

    @property
    def input_precision(self):
        return self.input_casts.input_cast.get_precision()

    @property
    def weight_precision(self):
        return self.weight_cast.get_precision()

    @property
    def weight_storage_precision(self):
        return self.weight_storage_cast.get_precision()

    @property
    def weight_scale(self):
        return self.weight_cast.scale.to(self.weight.device)

    @property
    def weight_zero_point(self):
        return self.weight_cast.zero_point.to(self.weight.device)

    @property
    def weight_storage_scale(self):
        return self.weight_storage_cast.scale.to(self.weight.device)

    @property
    def weight_storage_zero_point(self):
        return self.weight_storage_cast.zero_point.to(self.weight.device)

    @property
    def output_formats(self):
        return {k: cast.format for k, cast in self.output_casts.items()}

    @property
    def accum_format(self):
        return (self.accum_cast.format) if self.accum_cast is not None else None

    @property
    def weight_format(self):
        return (self.weight_cast.format) if self.weight_cast is not None else None

    @property
    def weight_storage_format(self):
        return (
            (self.weight_storage_cast.format)
            if self.weight_storage_cast is not None
            else None
        )

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
