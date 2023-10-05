from typing import Optional
import torch
import numpy as np
import transformers
import warnings
from contextlib import contextmanager


def _count_conv_flops(
    layer: torch.nn.modules.conv._ConvNd,
    _input: torch.Tensor,
    _output: torch.Tensor,
) -> int:
    r"""
    Taken from https://github.com/sovrasov/flops-counter.pytorch/blob/master/ptflops/pytorch_ops.py
    """
    batch_size = _input.shape[0]
    output_dims = list(_output.shape[2:])
    kernel_dims = list(layer.kernel_size)
    in_channels = layer.in_channels
    out_channels = layer.out_channels
    groups = layer.groups
    filters_per_channel = out_channels // groups
    conv_per_position_flops = (
        int(np.prod(kernel_dims)) * in_channels * filters_per_channel
    )
    active_elements_count = batch_size * int(np.prod(output_dims))
    return conv_per_position_flops * active_elements_count


def _count_linear_flops(
    layer: torch.nn.Linear,
    _input: torch.Tensor,
    _output: torch.Tensor,
) -> int:
    return int(np.prod(list(_input.shape))) * layer.out_features


def _count_hf_transformers_conv1d_flops(
    layer: transformers.pytorch_utils.Conv1D,
    _input: torch.Tensor,
    _output: torch.Tensor,
) -> int:
    return int(np.prod(list(_input.shape))) * layer.nf


class PerformanceProxyMixin:
    flop_counter: Optional[int] = None
    flop_counter_enabled: bool = False
    last_input_shape: Optional[torch.Size] = None
    last_output_shape: Optional[torch.Size] = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def zero_flop_counter(self) -> None:
        self.flop_counter = 0

    def enable_flop_counter(self, state: bool = True) -> None:
        self.flop_counter_enabled = state
        if self.flop_counter_enabled and self.flop_counter is None:
            self.zero_flop_counter()

    def count_flops(
        self,
        _input: torch.Tensor,
        _output: torch.Tensor,
    ) -> None:
        if self.flop_counter is not None:
            self.last_input_shape = _input.shape
            self.last_output_shape = _output.shape
            if isinstance(self, torch.nn.modules.conv._ConvNd):
                self.flop_counter += _count_conv_flops(self, _input, _output)
            elif isinstance(self, torch.nn.Linear):
                self.flop_counter += _count_linear_flops(self, _input, _output)
            elif isinstance(self, transformers.pytorch_utils.Conv1D):
                self.flop_counter += _count_hf_transformers_conv1d_flops(
                    self, _input, _output
                )
            else:
                self.flop_counter = None
                warnings.warn(
                    f"cannot count flops of this kind of layer: {type(self)}",
                    RuntimeWarning,
                )

    def _has_weight(self) -> bool:
        return "weight" in [n for n, _ in self.named_parameters()]

    @property
    def weight_elem_count(self) -> Optional[float]:
        if self._has_weight():
            _n = float(self.weight.nelement())
            if self.weight_sparsifier is not None:
                _n *= (
                    self.weight_sparsifier.density
                )  # not considering compression scheme and the overhead coming with different schemes
            return _n
        else:
            return None

    @property
    def weight_size_in_bytes(self) -> Optional[float]:
        if self._has_weight():
            if self.weight_cast is not None:
                bytes_per_elem = (
                    self.weight.element_size()
                    if self.weight_cast.format.bytes_per_elem is None
                    else self.weight_cast.format.bytes_per_elem
                )
            return bytes_per_elem * self.weight_elem_count
        else:
            return None

    @property
    def flops(self) -> Optional[float]:
        _flops = self.flop_counter
        if (
            _flops is not None
            and self._has_weight()
            and self.weight_sparsifier is not None
        ):
            _flops *= self.weight_sparsifier.density
        return _flops

    @property
    def bops(self) -> Optional[float]:
        _bops = self.flops
        if _bops is not None and self._has_weight():
            _bops *= self.input_precision * self.weight_precision
        return _bops

    @contextmanager
    def counting_flops(self, zero: bool = True) -> None:
        self.enable_flop_counter(True)
        if zero:
            self.zero_flop_counter()
        yield self
        self.enable_flop_counter(False)
