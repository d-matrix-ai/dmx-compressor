from mltools.numerical.observer import (DMXObserverBase,
                                        DummyObserver,
                                        MinMaxObserver,
                                        HistogramObserver,
                                        PercentileObserver,)

from collections import namedtuple
from typing import Dict, List, Optional, Any, Union, Type, Tuple

from mltools.numerical.format import Format, BlockFloatingPoint

import re
import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from functools import partial
import warnings

default_bfp_format = BlockFloatingPoint.from_shorthand("BFP[8|8]{64}(SN)")

default_xp_format = Format.from_shorthand("XP[8,0](CSN)")

from torch.ao.quantization.utils import (
    check_min_max_valid, calculate_qmin_qmax, is_per_tensor, is_per_channel, validate_qmin_qmax)

class DMXObserverBase(DMXObserverBase):
    '''
     Holds the d-Matrix DMXObserver implementation
    '''
    pass

class HistogramObserver(HistogramObserver):
    '''
    Holds the d-Matrix Histogram Observer implementation
    '''
    pass

class PercentileObserver(PercentileObserver):
    '''
    Holds the d-Matrix Percentile Observer implementation
    '''
    pass

class MinMaxObserver(MinMaxObserver):
    '''
    Holds the MinMaxObserver class of DMX
    '''
    pass

class _PartialWrapper:
    def __init__(self, p):
        self.p = p
        self.callable_args = {}

    def __call__(self, *args, **keywords):
        # call each arg in callable_args and add them partial, then run with keywords
        # skip if arg_name in keywords so its possible to overwrite
        for arg_name in self.callable_args:
            if arg_name not in keywords:
                keywords = {**keywords, arg_name: self.callable_args[arg_name]()}
        return self.p(*args, **keywords)

    def __repr__(self):
        return self.p.__repr__() + self.callable_args.__repr__()

    def with_args(self, **kwargs):
        return _with_args(self, **kwargs)

    def with_callable_args(self, **kwargs):
        result = _PartialWrapper(p=self.p)
        result.callable_args = {**self.callable_args, **kwargs}
        return result
    
def _with_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances. Can be used in conjunction with
    _callable_args

    Example::

        >>> # xdoctest: +SKIP("Undefined vars")
        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1) == id(foo_instance2)
        False
    """
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r


class PlaceholderObserver(DMXObserverBase):
    r"""
    Observer that doesn't do anything and just passes its configuration to the
    quantized module's ``.from_float()``.

    Can be used for quantization to float16 which doesn't require determining
    ranges.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        quant_min: minimum value in quantized domain (TODO: align behavior with other observers)
        quant_max: maximum value in quantized domain
        custom_op_name: (temporary) specify this observer for an operator that doesn't require any observation
                        (Can be used in Graph Mode Passes for special case ops).
        compute_dtype (deprecated): if set, marks the future quantize function to use
                       dynamic quantization instead of static quantization.
                       This field is deprecated, use `is_dynamic=True` instead.
        is_dynamic: if True, the `quantize` function in the reference model
                    representation taking stats from this observer instance will
                    use dynamic quantization.
    """

    def __init__(
        self, dtype=Format, custom_op_name="", compute_dtype=None,
        quant_min=None, quant_max=None, qscheme=None, eps=None,
        is_dynamic=False,
    ) -> None:
        super().__init__(dtype=dtype, is_dynamic=is_dynamic)
        if qscheme is None:
            qscheme = torch.per_tensor_affine
        if eps is None:
            eps = torch.finfo(torch.float32).eps

        # dtype of input of the target operator, e.g. for dynamic quantization
        # ops, the dtype will be float32
        self.dtype = dtype
        self.qscheme = qscheme
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = torch.Tensor([eps])
        self.custom_op = custom_op_name
        # used for configuration of computation type for dynamic quantization
        if compute_dtype:
            is_dynamic = True
            warnings.warn(
                "Please use `is_dynamic` instead of `compute_dtype`. \
                    `compute_dtype` will be deprecated in a future release \
                    of PyTorch."
            )

    def forward(self, x):
        return x

    @torch.jit.export
    def extra_repr(self):
        return f"dtype={self.dtype}, is_dynamic={self.is_dynamic}"

    @torch.jit.export
    def calculate_qparams(self):
        raise Exception(  # noqa: TRY002
            "calculate_qparams should not be called for PlaceholderObserver"
        )


class RecordingObserver(DMXObserverBase):
    r"""
    The module is mainly for debug and records the tensor values during runtime.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
    """
    __annotations__ = {"tensor_val": List[Optional[torch.Tensor]]}

    def __init__(self, dtype: Format):
        super().__init__(dtype=dtype, is_dynamic=False)  # type: ignore[call-arg]
        self.tensor_val = []

    def forward(self, x):
        self.tensor_val.append(x.clone())
        return x

    @torch.jit.export
    def calculate_qparams(self):
        raise Exception("calculate_qparams should not be called for RecordingObserver")  # noqa: TRY002

    @torch.jit.export
    def get_tensor_value(self):
        return self.tensor_val
    
class UniformQuantizationObserverBase(DMXObserverBase):
    r"""Common base for all observers using uniform quantization to calculate
    scale and zero_point.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used.
        reduce_range: Reduces the range of the quantized data type by 1 bit.
                      This is sometimes required to avoid instruction overflow.
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    .. warning::

        :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.
               or `torch.int8` or `torch.uint8`

    .. warning::

        :attr:`qscheme` can only take one of the following options:

        - ``torch.per_tensor_affine``
        - ``torch.per_tensor_symmetric``
        - ``torch.per_channel_affine``
        - ``torch.per_channel_symmetric``
    """

    # Note: the version is shared by all observer types
    #
    # Version 1/None
    #   self
    #
    # Version 2 (base class only, does not include child class buffers)
    #   self
    #   |--- eps : Tensor
    #
    # Version 3
    #   for HistogramObserver only, changed the shape of uninitialized
    #   min_val and max_val buffers from torch.Size([0]) to torch.Size([])
    #   for PerChannelObservers, changed the name of the buffers from min_vals
    #   to min_val and from max_vals to max_val.
    _version = 3

    eps: torch.Tensor

    def __init__(
        self,
        dtype=default_xp_format,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        super().__init__(dtype=dtype, is_dynamic=is_dynamic, **kwargs)
        self.qscheme = qscheme
        if reduce_range:
            warnings.warn(
                "Please use quant_min and quant_max to specify the range for observers. \
                    reduce_range will be deprecated in a future release of PyTorch."
            )
        self.reduce_range = reduce_range
        self.register_buffer(
            "eps", torch.tensor([eps], **factory_kwargs)
        )
        assert self.qscheme in (
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
            torch.per_channel_affine,
            torch.per_channel_symmetric,
            torch.per_channel_affine_float_qparams,
        ), "Default Observer only works for per_tensor_affine, \
                per_tensor_symmetric, per_channel_affine, \
                per_channel_symmetric and per_channel_float_qparams quantization scheme"

        _ALLOWED_DTYPES = (
            Format,
            torch.float32,
        )
        assert isinstance(self.dtype, Format), f"Default Observer only works for {_ALLOWED_DTYPES} data type"
        # assert self.dtype in _ALLOWED_DTYPES, f"Default Observer only works for {_ALLOWED_DTYPES} data type"
        self.has_customized_qrange = (quant_min is not None) and (quant_max is not None)
        if self.has_customized_qrange:
            validate_qmin_qmax(quant_min, quant_max)
        self.quant_min, self.quant_max = \
            calculate_qmin_qmax(quant_min, quant_max, self.has_customized_qrange, self.dtype, self.reduce_range)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):

        version = local_metadata.get("version", None)

        if version is None or version == 1:
            # eps was moved to a buffer in version 2
            eps = torch.tensor([torch.finfo(torch.float32).eps])
            state_dict[prefix + "eps"] = eps

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def _validate_qmin_qmax(self, quant_min: int, quant_max: int) -> None:
        r"""Validates that the user-specified quantization range is properly initialized
        and within the given bound supported by the observer dtype.

        To accommodate lower-bit quantization with respect to the existing torch.qint8 and
        torch.quint8 datatypes, the user can choose to use dynamic quantization range by passing
        in a tuple of initial qmin and qmax values. One use case is these customized qmin and qmax
        values are used to calculate static estimates of the scale and zero point for aggressive lower-bit
        fake quantization. These estimates are compared against parameters learned through backpropagation.
        The related literatures for scale and zero point via backpropagation are as follows:

        Learned Step Size Quantization: https://openreview.net/pdf?id=rkgO66VKDS
        Trained Quantization Thresholds: https://arxiv.org/pdf/1903.08066.pdf
        """
        # The variable names are prefixed with "initial" because their values (qmin and qmax) might be adjusted
        # based on whether quantization range is reduced and the datatype (signed/unsigned) used by the observer.
        assert (
            quant_min <= 0 <= quant_max
        ), "Used-specified quantization range must include 0."
        assert (
            quant_min < quant_max
        ), "qmin must be strictly less than qmax for user-specified quantization range."

    @torch.jit.export
    def _calculate_qparams(
        self, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters, given min and max
        value tensors. Works for both per tensor and per channel cases

        Args:
            min_val: Minimum values per channel
            max_val: Maximum values per channel

        Returns:
            scales: Scales tensor of shape (#channels,)
            zero_points: Zero points tensor of shape (#channels,)
        """
        # Functionally equivalent to 'determine_qparams' in utils.py. Observers must be torchscriptable however and qscheme
        # as far as I can tell is not allowed to passed as a parameter in torchscript functions. This makes refactoring observer
        # to use this utility a massive pain and very gross. For now Im opting just to duplicate as this code
        # seems unlikey to change (last update over 1 year ago) and when torchscript is fully deprecated we can refactor.
        # TODO(jakeszwe, jerryzh168)
        if not check_min_max_valid(min_val, max_val):
            return torch.tensor([1.0], device=min_val.device.type), torch.tensor([0], device=min_val.device.type)

        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

        if (
            self.qscheme == torch.per_tensor_symmetric
            or self.qscheme == torch.per_channel_symmetric
        ):
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
            if self.dtype in [torch.quint8, torch.uint8]:
                if self.has_customized_qrange:
                    # When customized quantization range is used, down-rounded midpoint of the range is chosen.
                    zero_point = zero_point.new_full(
                        zero_point.size(), (quant_min + quant_max) // 2
                    )
                else:
                    zero_point = zero_point.new_full(zero_point.size(), 128)
        elif self.qscheme == torch.per_channel_affine_float_qparams:
            scale = (max_val - min_val) / float(quant_max - quant_min)
            scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))
            # We use the quantize function
            # xq = Round(Xf * inv_scale + zero_point),
            # setting zero_point to (-1 * min *inv_scale) we get
            # Xq = Round((Xf - min) * inv_scale)
            zero_point = -1 * min_val / scale
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)

        # For scalar values, cast them to Tensors of size 1 to keep the shape
        # consistent with default values in FakeQuantize.
        if len(scale.shape) == 0:
            # TODO: switch to scale.item() after adding JIT support
            scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
        if len(zero_point.shape) == 0:
            # TODO: switch to zero_point.item() after adding JIT support
            zero_point = torch.tensor(
                [int(zero_point)], dtype=zero_point.dtype, device=device
            )
            if self.qscheme == torch.per_channel_affine_float_qparams:
                zero_point = torch.tensor(
                    [float(zero_point)], dtype=zero_point.dtype, device=device
                )

        return scale, zero_point

    @torch.jit.export
    def reset_min_max_vals(self):
        raise NotImplementedError("Cannot reset min/max values in the given observer.")


# Originally, this class was called `_ObserverBase`.  Keeping the old name around
# for backwards compatibility.
# TODO(after v1.13): delete this
_ObserverBase = UniformQuantizationObserverBase
    
class PerChannelMinMaxObserver(UniformQuantizationObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        ch_axis: Channel axis
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The quantization parameters are computed the same way as in
    :class:`~torch.ao.quantization.observer.MinMaxObserver`, with the difference
    that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        ch_axis=0,
        dtype=default_xp_format,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        if not is_per_channel(qscheme):
            raise NotImplementedError(
                "PerChannelMinMaxObserver's qscheme only support \
                    torch.per_channel_symmetric, torch.per_channel_affine and torch.per_channel_affine_float_qparams."
            )
        if is_dynamic:
            raise NotImplementedError(
                "PerChannelMinMaxObserver doesn't support dynamic quantization"
            )
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.ch_axis = ch_axis
        self.register_buffer("min_val", torch.tensor([], **factory_kwargs))
        self.register_buffer("max_val", torch.tensor([], **factory_kwargs))
        if (
            self.qscheme == torch.per_channel_symmetric
            and self.reduce_range
            and self.dtype == torch.quint8
        ):
            raise NotImplementedError(
                "Cannot reduce range for symmetric quantization for quint8"
            )

    def forward(self, x_orig):
        return self._forward(x_orig)

    def _forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        y = y.to(self.min_val.dtype)
        y = torch.flatten(y, start_dim=1)
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = torch.aminmax(y, dim=1)
        else:
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
            min_val = torch.min(min_val_cur, min_val)
            max_val = torch.max(max_val_cur, max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        return self._calculate_qparams(self.min_val, self.max_val)

    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}"

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, torch.Tensor],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        version = local_metadata.get("version", None)
        if version is not None and version < 3:
            local_state = ["min_vals", "max_vals"]
            expected_min_name = "min_vals"
            expected_max_name = "max_vals"
        else:
            local_state = ["min_val", "max_val"]
            expected_min_name = "min_val"
            expected_max_name = "max_val"
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading min_val or max_val
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == expected_min_name:
                    self.min_val.resize_(val.shape)
                elif name == expected_max_name:
                    self.max_val.resize_(val.shape)
                else:
                    warnings.warn(f"Observer load_from_state_dict got unexpected name {name}")
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == expected_min_name:
                        self.min_val.copy_(val)
                    elif name == expected_max_name:
                        self.max_val.copy_(val)
                    else:
                        warnings.warn(f"Observer load_from_state_dict got unexpected name {name}")
            elif strict:
                missing_keys.append(key)

        if not torch.jit.is_scripting():
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                False,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

    def _load_from_state_dict_script(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, torch.Tensor],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):

        self._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        # This used to be torch.ones but that does not work because
        # JIT compiler can optimize it via common subexpression elimination
        # in which case both min_val and max_val point to the same tensor.
        self.min_val = torch.rand(0, )
        self.max_val = torch.rand(0, )

class MovingAverageMinMaxObserver(MinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    moving average of the min and max values.

    This observer computes the quantization parameters based on the moving
    averages of minimums and maximums of the incoming tensors. The module
    records the average minimum and maximum of incoming tensors, and uses this
    statistic to compute the quantization parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The moving average min/max is computed as follows

    .. math::

        \begin{array}{ll}
                x_\text{min} = \begin{cases}
                    \min(X) & \text{if~}x_\text{min} = \text{None} \\
                    (1 - c) x_\text{min} + c \min(X) & \text{otherwise}
                \end{cases}\\
                x_\text{max} = \begin{cases}
                    \max(X) & \text{if~}x_\text{max} = \text{None} \\
                    (1 - c) x_\text{max} + c \max(X) & \text{otherwise}
                \end{cases}\\
        \end{array}

    where :math:`x_\text{min/max}` is the running average min/max, :math:`X` is
    is the incoming tensor, and :math:`c` is the ``averaging_constant``.

    The scale and zero point are then computed as in
    :class:`~torch.ao.quantization.observer.MinMaxObserver`.

    .. note:: Only works with ``torch.per_tensor_affine`` quantization scheme.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """

    def __init__(
        self,
        averaging_constant=0.01,
        dtype=default_xp_format,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs
    ) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                f"MovingAverageMinMaxObserver's qscheme only support \
                torch.per_tensor_symmetric and torch.per_tensor_affine. \
                but got: {qscheme}"
            )
        self.averaging_constant = averaging_constant
        if is_dynamic and self.averaging_constant != 1:
            raise NotImplementedError(
                "MovingAverageMinMaxObserver doesn't support dynamic quantization for "
                f"averaging constant of {self.averaging_constant}"
            )
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs
        )

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        if min_val == float("inf") and max_val == float("-inf"):
            min_val, max_val = torch.aminmax(x)
        else:
            min_val_cur, max_val_cur = torch.aminmax(x)
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig


class MovingAveragePerChannelMinMaxObserver(PerChannelMinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        ch_axis: Channel axis
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The quantization parameters are computed the same way as in
    :class:`~torch.ao.quantization.observer.MovingAverageMinMaxObserver`, with the
    difference that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """

    def __init__(
        self,
        averaging_constant=0.01,
        ch_axis=0,
        dtype=default_xp_format,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs
    ) -> None:
        if not is_per_channel(qscheme):
            raise NotImplementedError(
                "MovingAveragePerChannelMinMaxObserver's qscheme only support \
                    torch.per_channel_symmetric, torch.per_channel_affine and torch.per_channel_affine_float_qparams."
            )
        if is_dynamic:
            raise NotImplementedError(
                "MovingAveragePerChannelMinMaxObserver doesn't support dynamic quantization"
            )
        super().__init__(
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs
        )
        self.averaging_constant = averaging_constant


class FixedQParamsObserver(DMXObserverBase):
    r"""
    Observer that simulates quantize and dequantize with fixed
    quantization parameters in training time. Only per tensor
    quantization is supported.

    Args:
        `scale` (float): fixed scale for the observer
        `zero_point` (int): fixed zero point for the observer
        `dtype`, `qscheme`, `quant_min`, `quant_max`
    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(
        self,
        scale,
        zero_point,
        dtype=Format,
        qscheme=torch.per_tensor_affine,
        quant_min=0,
        quant_max=255,
        is_dynamic=False,
        **kwargs,
    ):
        if is_dynamic:
            raise NotImplementedError(
                "FixedQParamsObserver doesn't support dynamic quantization"
            )
        super().__init__(dtype=dtype, is_dynamic=is_dynamic, **kwargs)
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.register_buffer('scale', torch.tensor([scale], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([zero_point], dtype=torch.int))
        self.dtype = dtype
        self.qscheme = qscheme

    def forward(self, X):
        return X

    @torch.jit.export
    def calculate_qparams(self):
        return self.scale, self.zero_point

class ReuseInputObserver(DMXObserverBase):
    r""" This observer is used when we want to reuse the observer from the operator
    that produces the input Tensor, typically used for operators like reshape, e.g.
    ```
    x0 = ...
    x1 = x0.reshape()
    ```
    if we configure x0 to be observed by some observer, let's say MinMaxObserver,
    and reshape is configured with ReuseInputObserver, we'll reuse the observer instance
    for x0 for x1 (output of reshape). If x0 is not observed, we also won't observe x1.

    Note: this is only enabled in FX Graph Mode Quantization
    """
    def __init__(self):
        super().__init__(Format, is_dynamic=False)

    def forward(self, x):
        return x

    @torch.jit.export
    def calculate_qparams(self):
        raise Exception("calculate_qparams should not be called for ReuseInputObserver")  # noqa: TRY002

class NoopObserver(DMXObserverBase):
    r"""
    Observer that doesn't do anything and just passes its configuration to the
    quantized module's ``.from_float()``.

    Primarily used for quantization to float16 which doesn't require determining
    ranges.

    Args:
        dtype: Quantized data type
        custom_op_name: (temporary) specify this observer for an operator that doesn't require any observation
                        (Can be used in Graph Mode Passes for special case ops).
    """

    def __init__(self, dtype=torch.float16, custom_op_name="") -> None:
        super().__init__(dtype=dtype, is_dynamic=False)
        self.dtype = dtype
        self.custom_op = custom_op_name

    def forward(self, x):
        return x

    @torch.jit.export
    def calculate_qparams(self):
        raise Exception("calculate_qparams should not be called for NoopObserver")  # noqa: TRY002
 
def _is_observer_script_module(mod, obs_type_name):
    """Returns true if given mod is an instance of Observer script module."""
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        # qualified name looks like '__torch__.torch.ao.quantization.observer.___torch_mangle_2.MinMaxObserver'
        suffix = mod._c.qualified_name.split(".", 1)[1]
        name = re.sub(r"\.___torch_mangle_\d+", "", suffix)
        return obs_type_name in name
    return False


def _is_activation_post_process(module):
    return (
        isinstance(module, (torch.ao.quantization.ObserverBase,
                            torch.ao.quantization.FakeQuantizeBase)) or _is_observer_script_module(module, "quantization.observer")
    )

# Restrict activations to be in the range (0,127)
default_observer = MinMaxObserver.with_args(quant_min=0, quant_max=127)
"""
Default observer for static quantization, usually used for debugging.
"""

default_placeholder_observer = DummyObserver
"""
Default placeholder observer, usually used for quantization to torch.float16.
"""

default_debug_observer = RecordingObserver
"""
Default debug-only observer.
"""

default_weight_observer = MinMaxObserver.with_args(
    dtype=default_xp_format, qscheme=torch.per_tensor_symmetric
)
"""
Default weight observer.
"""

default_per_channel_weight_observer = PerChannelMinMaxObserver.with_args(
    dtype=default_xp_format, qscheme=torch.per_channel_symmetric
)
"""
Default per-channel weight observer, usually used on backends where per-channel
weight quantization is supported, such as `fbgemm`.
"""

default_dynamic_quant_observer = PlaceholderObserver.with_args(
    dtype=default_bfp_format, quant_min=0, quant_max=255, is_dynamic=True,
)
"""
Default observer for dynamic quantization.
"""

default_placeholder_observer = PlaceholderObserver
"""
Default placeholder observer, usually used for quantization to torch.float16.
"""

default_float_qparams_observer = PerChannelMinMaxObserver.with_args(
    dtype=default_xp_format, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0
)
"""
Default observer for a floating point zero-point.
"""

default_float_qparams_observer_4bit = PerChannelMinMaxObserver.with_args(
    dtype=default_xp_format, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0
)
"""
Default observer for a floating point zero-point and 4 bit activations.
"""

default_reuse_input_observer = ReuseInputObserver
"""
Default observer for operators like reshape that reuses the observer of input to
the operator
"""

weight_observer_range_neg_127_to_127 = MinMaxObserver.with_args(
    dtype=default_xp_format, qscheme=torch.per_tensor_symmetric,
    quant_min=-127, quant_max=127, eps=2 ** -12)
"""
Symmetric weight observer with the 8-bit values restricted to [-127, +127], excluding -128.
"""
default_fixed_qparams_range_neg1to1_observer = FixedQParamsObserver.with_args(
    scale=2.0 / 256.0, zero_point=128, dtype=default_bfp_format, quant_min=0, quant_max=255)
default_fixed_qparams_range_0to1_observer = FixedQParamsObserver.with_args(
    scale=1.0 / 256.0, zero_point=0, dtype=default_bfp_format, quant_min=0, quant_max=255)
