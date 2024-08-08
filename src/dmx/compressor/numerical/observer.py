import torch
from typing import Optional, Tuple, Dict
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.utils import (
    check_min_max_valid,
    is_per_tensor,
    is_per_channel,
)
import numpy as np
from .format import Format, FixedPoint


def _get_qmin_qmax(dtype: Format) -> Tuple[int, int]:
    if isinstance(dtype, FixedPoint) and dtype.fraction == 0 and dtype.clamp:
        quant_min = -(2 ** (dtype.precision - 1))
        quant_max = 2 ** (dtype.precision - 1) - 1
        if dtype.symmetric:
            quant_min += 1
    else:
        quant_min, quant_max = None, None
    return quant_min, quant_max


class DMXObserverBase(ObserverBase):
    r"""
    Taken from torch.ao.quantization.observer.UniformQuantizationObserverBase
    """
    eps: torch.Tensor

    def __init__(
        self,
        dtype: Format,
        qscheme: torch.qscheme = torch.per_tensor_affine,
        factory_kwargs: Optional[dict] = None,
        eps: float = torch.finfo(torch.float32).eps,
        **kwargs,
    ) -> None:
        if dtype == torch.float32:
            dtype = Format.from_shorthand("SAME")
        assert isinstance(dtype, Format), f"illegal format {dtype}"
        super().__init__(dtype=dtype, **kwargs)
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.qscheme = qscheme
        assert self.qscheme in (
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
            torch.per_channel_affine,
            torch.per_channel_symmetric,
            torch.per_channel_affine_float_qparams,
        ), "Default Observer only works for per_tensor_affine, \
                per_tensor_symmetric, per_channel_affine, \
                per_channel_symmetric and per_channel_float_qparams quantization scheme"
        self.register_buffer("eps", torch.tensor([eps], **factory_kwargs))
        assert isinstance(
            self.dtype, Format
        ), "Default DMXObserver only works for dmx.compressor.numerical.Format objects as dtype"
        self.quant_min, self.quant_max = _get_qmin_qmax(self.dtype)

    @torch.jit.export
    def _calculate_qparams(
        self, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters, given min and max
        value tensors. Works for both per tensor and per channel cases
        """
        if not check_min_max_valid(min_val, max_val):
            return torch.tensor([1.0], device=min_val.device.type), torch.tensor(
                [0], device=min_val.device.type
            )

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
            scale = torch.max(scale, self.eps.to(device))
        elif self.qscheme == torch.per_channel_affine_float_qparams:
            scale = (max_val - min_val) / float(quant_max - quant_min)
            self.eps = self.eps.to(device)
            scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))
            # We use the quantize function
            # xq = Round(Xf * inv_scale + zero_point),
            # setting zero_point to (-1 * min *inv_scale) we get
            # Xq = Round((Xf - min) * inv_scale)
            zero_point = -1 * min_val / scale
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps.to(device))
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

    def extra_repr(self):
        return f"quant_min = {self.quant_min}, quant_max = {self.quant_max}"


class DummyObserver(DMXObserverBase):
    r"""
    This is a dummy observer that does not do anything
    """

    def __init__(self, dtype: Format, ch_axis: int = -1, **kwargs) -> None:
        super().__init__(dtype=dtype, **kwargs)
        self.dtype = dtype
        self.ch_axis = ch_axis

    def forward(self, x):
        return x

    @torch.jit.export
    def calculate_qparams(self):
        return self._calculate_qparams(torch.empty(0), torch.empty(0))


class MinMaxObserver(DMXObserverBase):
    r"""
    Adapted from torch.ao.quantization.observer.MinMaxObserver,
    supports per-channel
    """
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        dtype: Format = Format.from_shorthand("XP[8,0](CSN)"),
        qscheme: Optional[torch.qscheme] = torch.per_tensor_affine,
        ch_axis: int = -1,
        factory_kwargs: Optional[Dict] = None,
        eps: Optional[float] = torch.finfo(torch.float32).eps,
        **kwargs,
    ) -> None:
        if qscheme == torch.per_channel_affine_float_qparams:
            raise NotImplementedError(
                "MinMaxObserver does not support qscheme: torch.per_channel_affine_float_qparams"
            )
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            factory_kwargs=factory_kwargs,
            eps=eps,
            **kwargs,
        )
        self.ch_axis = ch_axis
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        dims = list(range(x.dim()))
        x = x.to(self.min_val.dtype)
        if is_per_channel(self.qscheme):
            dims.pop(self.ch_axis)
        min_val_cur = torch.amin(x, dim=dims)
        max_val_cur = torch.amax(x, dim=dims)
        self.min_val = self.min_val.to(min_val_cur.device)
        self.max_val = self.max_val.to(max_val_cur.device)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        if self.min_val.shape:
            self.min_val.copy_(min_val)
            self.max_val.copy_(max_val)
        else:
            self.min_val.data = min_val
            self.max_val.data = max_val
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""
        return self._calculate_qparams(self.min_val, self.max_val)

    @torch.jit.export
    def extra_repr(self):
        return super().extra_repr() + ", min_val = {}, max_val = {}".format(
            self.min_val, self.max_val
        )

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val.copy_(torch.tensor(float("inf")))
        self.max_val.copy_(torch.tensor(float("-inf")))


class HistogramObserver(DMXObserverBase):
    r"""
    Adapted from torch.ao.quantization.observer.HistogramObserver,
    still does not support per-channel
    """
    histogram: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        bins: int = 2048,
        upsample_rate: int = 128,
        dtype: Format = Format.from_shorthand("XP[8,0](CSN)"),
        qscheme: Optional[torch.qscheme] = torch.per_tensor_affine,
        ch_axis: int = -1,  # TODO: implement for per-channel
        factory_kwargs: Optional[dict] = None,
        eps=torch.finfo(torch.float32).eps,
        **kwargs,
    ) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                "HistogramObserver's qscheme only support torch.per_tensor_symmetric or torch.per_tensor_affine."
            )
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            factory_kwargs=factory_kwargs,
            eps=eps,
            **kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.bins = bins
        self.register_buffer("histogram", torch.zeros(self.bins, **factory_kwargs))
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        self.upsample_rate = upsample_rate

    def _get_norm(
        self,
        delta_begin: torch.Tensor,
        delta_end: torch.Tensor,
        density: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end.
        Currently only L2 norm is supported.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        norm = (
            delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin
        ) / 3
        return density * norm

    def _compute_quantization_error(self, next_start_bin: int, next_end_bin: int):
        r"""
        Compute the quantization error if we use start_bin to end_bin as the
        min and max to do the quantization.
        """
        dst_nbins = 2**self.dtype.precision
        bin_width = (self.max_val.item() - self.min_val.item()) / self.bins

        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        src_bin = torch.arange(self.bins, device=self.histogram.device)
        # distances from the beginning of first dst_bin to the beginning and
        # end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dst_bins the beginning and end of src_bin belong to?
        dst_bin_of_begin = torch.clamp(
            torch.div(src_bin_begin, dst_bin_width, rounding_mode="floor"),
            0,
            dst_nbins - 1,
        )
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = torch.clamp(
            torch.div(src_bin_end, dst_bin_width, rounding_mode="floor"),
            0,
            dst_nbins - 1,
        )
        dst_bin_of_end_center = (dst_bin_of_end + 0.5) * dst_bin_width

        density = self.histogram / bin_width

        norm = torch.zeros(self.bins, device=self.histogram.device)

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self._get_norm(
            delta_begin,
            torch.ones(self.bins, device=self.histogram.device) * delta_end,
            density,
        )

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(
            torch.tensor(-dst_bin_width / 2), torch.tensor(dst_bin_width / 2), density
        )

        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2

        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(torch.tensor(delta_begin), delta_end, density)

        return norm.sum().item()

    def _non_linear_param_search(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
        assert self.histogram.size()[0] == self.bins, "bins mistmatch"
        bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = torch.sum(self.histogram).item()
        cSum = torch.cumsum(self.histogram, dim=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")

        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize

            # find the left and right bins between the quantile bounds
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            # decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) > (end_bin - r):
                # move the start bin
                next_start_bin = l
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = r
                beta = next_beta

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = self._compute_quantization_error(next_start_bin, next_end_bin)

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max

    def _adjust_min_max(
        self, combined_min: torch.Tensor, combined_max: torch.Tensor, upsample_rate: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        # We ensure that:
        # (combined_max - combined_min)/(downsample_rate*Nbins) = (max - min)/(upsample_rate*Nbins)
        # This allows us to have a common grid of resolution s, where we can align
        # the input histogram
        # start_idx maps min_val to the histogram bin index.

        hist_bin_width = (self.max_val - self.min_val) / (self.bins * upsample_rate)
        downsample_rate = int(
            torch.ceil(
                (combined_max - combined_min) / (self.bins * hist_bin_width)
            ).item()
        )
        e = downsample_rate * (self.bins * hist_bin_width) - (
            combined_max - combined_min
        )
        # Relax only the max, not the min, so that for one sided distributions, min stays at zero
        combined_max = combined_max + e
        combined_min = combined_min
        start_idx = int(
            torch.round((self.min_val - combined_min) / hist_bin_width).item()
        )
        return combined_min, combined_max, downsample_rate, start_idx

    def _combine_histograms(
        self,
        orig_hist: torch.Tensor,
        new_hist: torch.Tensor,
        upsample_rate: int,
        downsample_rate: int,
        start_idx: int,
        Nbins: int,
    ) -> torch.Tensor:
        # First up-sample the histogram with new data by a factor of L
        # This creates an approximate probability density thats piecwise constant
        upsampled_histogram = new_hist.repeat_interleave(upsample_rate)
        # Now insert the upsampled histogram into the output
        # histogram, which is initialized with zeros.
        # The offset at which the histogram is introduced is determined
        # by the start index as the output histogram can cover a wider range
        histogram_with_output_range = torch.zeros(
            (Nbins * downsample_rate), device=orig_hist.device
        )
        histogram_with_output_range[
            start_idx : Nbins * upsample_rate + start_idx
        ] = upsampled_histogram
        # Compute integral histogram, double precision is needed to ensure
        # that there are no overflows
        integral_histogram = torch.cumsum(
            histogram_with_output_range, 0, dtype=torch.double
        )[downsample_rate - 1 :: downsample_rate]
        # Finally perform interpolation
        shifted_integral_histogram = torch.zeros((Nbins), device=orig_hist.device)
        shifted_integral_histogram[1:Nbins] = integral_histogram[0:-1]
        interpolated_histogram = (
            integral_histogram - shifted_integral_histogram
        ) / upsample_rate
        orig_hist = orig_hist + interpolated_histogram.to(torch.float)
        return orig_hist

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        min_val = self.min_val
        max_val = self.max_val
        same_values = min_val.item() == max_val.item()
        is_uninitialized = min_val == float("inf") and max_val == float("-inf")
        if is_uninitialized or same_values:
            min_val, max_val = torch.aminmax(x)
            self.min_val.resize_(min_val.shape)
            self.min_val.copy_(min_val)
            self.max_val.resize_(max_val.shape)
            self.max_val.copy_(max_val)
            assert (
                min_val.numel() == 1 and max_val.numel() == 1
            ), "histogram min/max values must be scalar."
            torch.histc(
                x, self.bins, min=int(min_val), max=int(max_val), out=self.histogram
            )
        else:
            new_min, new_max = torch.aminmax(x)
            combined_min = torch.min(new_min, min_val)
            combined_max = torch.max(new_max, max_val)
            # combine the existing histogram and new histogram into 1 histogram
            # We do this by first upsampling the histogram to a dense grid
            # and then downsampling the histogram efficiently
            (
                combined_min,
                combined_max,
                downsample_rate,
                start_idx,
            ) = self._adjust_min_max(combined_min, combined_max, self.upsample_rate)
            assert (
                combined_min.numel() == 1 and combined_max.numel() == 1
            ), "histogram min/max values must be scalar."
            combined_histogram = torch.histc(
                x, self.bins, min=int(combined_min), max=int(combined_max)
            )
            if combined_min == min_val and combined_max == max_val:
                combined_histogram += self.histogram
            else:
                combined_histogram = self._combine_histograms(
                    combined_histogram,
                    self.histogram,
                    self.upsample_rate,
                    downsample_rate,
                    start_idx,
                    self.bins,
                )

            self.histogram.detach_().resize_(combined_histogram.shape)
            self.histogram.copy_(combined_histogram)
            self.min_val.detach_().resize_(combined_min.shape)
            self.min_val.copy_(combined_min)
            self.max_val.detach_().resize_(combined_max.shape)
            self.max_val.copy_(combined_max)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        is_uninitialized = self.min_val == float("inf") and self.max_val == float(
            "-inf"
        )
        if is_uninitialized:
            return torch.tensor([1.0], device=self.min_val.device.type), torch.tensor(
                [0], device=self.min_val.device.type
            )
        assert self.bins == len(self.histogram), (
            "The number of bins in histogram should be equal to the number of bins "
            "supplied while making this observer"
        )

        new_min, new_max = self._non_linear_param_search()

        return self._calculate_qparams(new_min, new_max)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(HistogramObserver, self)._save_to_state_dict(
            destination, prefix, keep_vars
        )
        destination[prefix + "min_val"] = self.min_val
        destination[prefix + "max_val"] = self.max_val

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

        if version is None or version < 3:
            # if min_val and max_val are not initialized, update their shape
            # to account for the differences between v2 and v3
            min_val_name, max_val_name = prefix + "min_val", prefix + "max_val"
            if min_val_name in state_dict:
                if state_dict[min_val_name].shape == torch.Size([0]):
                    state_dict[min_val_name] = torch.tensor(float("inf"))
            if max_val_name in state_dict:
                if state_dict[max_val_name].shape == torch.Size([0]):
                    state_dict[max_val_name] = torch.tensor(float("-inf"))

        local_state = ["min_val", "max_val"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(HistogramObserver, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def extra_repr(self):
        return super().extra_repr() + ", min_val = {}, max_val = {}".format(
            self.min_val, self.max_val
        )


class PercentileObserver(HistogramObserver):
    r"""
    Extending HistogramObserver to allow calculation of percentile from histogram,
    taken from https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/pytorch_quantization/calib/histogram.py#L287
    """

    def __init__(
        self,
        percentile: float = 99.99,
        **kwargs,
    ) -> None:
        if percentile < 0 or percentile > 100:
            raise ValueError(
                "Invalid percentile. Must be in range 0 <= percentile <= 100."
            )
        self.percentile = percentile
        super().__init__(**kwargs)

    def _compute_amax_percentile(self):
        calib_bin_edges = []  # TODO: implement this
        total = self.histogram.sum()
        cdf = np.cumsum(self.histogram / total)
        idx = np.searchsorted(cdf, self.percentile / 100)
        calib_amax = calib_bin_edges[idx]
        calib_amax = torch.tensor(calib_amax.item())  # pylint: disable=not-callable
        return calib_amax

    @torch.jit.export
    def calculate_qparams(self):
        is_uninitialized = self.min_val == float("inf") and self.max_val == float(
            "-inf"
        )
        if is_uninitialized:
            return torch.tensor([1.0], device=self.min_val.device.type), torch.tensor(
                [0], device=self.min_val.device.type
            )
        assert self.bins == len(self.histogram), (
            "The number of bins in histogram should be equal to the number of bins "
            "supplied while making this observer"
        )

        new_min, new_max = self._non_linear_param_search()

        return self._calculate_qparams(new_min, new_max)

    @torch.jit.export
    def extra_repr(self):
        return "percentile = {}".format(self.min_val, )
