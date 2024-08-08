from dmx.compressor.numerical.observer import DMXObserverBase

from dmx.compressor.pt2bfp.fake_quantize import FakeQuantizeBase
from typing import Union, List, Callable, Tuple, Optional
from torch import Tensor
import torch

ObserverOrFakeQuantize = Union[DMXObserverBase, FakeQuantizeBase]
ObserverOrFakeQuantize.__module__ = "pt2bfp"

class _DerivedObserverOrFakeQuantize(DMXObserverBase):
    r"""This observer is used to describe an observer whose quantization parameters
    are derived from other observers
    """

    def __init__(
        self,
        dtype: torch.dtype,
        obs_or_fqs: List[ObserverOrFakeQuantize],
        derive_qparams_fn: Callable[[List[ObserverOrFakeQuantize]], Tuple[Tensor, Tensor]],
        quant_min: Optional[int]=None,
        quant_max: Optional[int]=None,
        qscheme: Optional[torch.qscheme]=None,
        ch_axis: Optional[int] = None
    ):
        super().__init__(dtype)
        self.obs_or_fqs = obs_or_fqs
        self.derive_qparams_fn = derive_qparams_fn
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.qscheme = qscheme
        self.ch_axis = ch_axis

        from .utils import is_per_channel
        if is_per_channel(self.qscheme):
            assert self.ch_axis is not None, "Must provide a valid ch_axis if qscheme is per channel"

    def forward(self, x: Tensor) -> Tensor:
        return x

    def calculate_qparams(self):
        return self.derive_qparams_fn(self.obs_or_fqs)