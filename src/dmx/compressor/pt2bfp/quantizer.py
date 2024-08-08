#!/usr/bin/env python3
#
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.ao.quantization import ObserverOrFakeQuantize
from dmx.compressor.pt2bfp.qconfig import _ObserverConstructor
from torch.fx import Node

from dmx.compressor.numerical.format import Format

from torch.ao.quantization.quantizer import QuantizationSpecBase

SUPPORTED_DTYPES = [Format]
SUPPORTED_QSCHEMES = [torch.per_tensor_affine, torch.per_tensor_symmetric, torch.per_channel_affine, torch.per_channel_symmetric]


class QuantizationSpecBase(QuantizationSpecBase):
    pass

@dataclass(eq=True, frozen=True)
class BFPQuantizationSpec(QuantizationSpecBase):
    """ Quantization spec for common operators that allows user to specify how to
    quantize a Tensor, this includes dtype, quant_min, quant_max etc.
    """
    dtype: Format = Format.from_shorthand("BFP[8|8]{64}(SN)")
    # observer or fake_quantize constructor such as
    # MinMaxObserver, PerChannelHistogramObserver etc.
    # or we can attach some custom args to them
    # e.g. MinMaxObserver.with_args(eps=eps)
    observer_or_fake_quant_ctr: _ObserverConstructor = None
    qscheme: Optional[torch.qscheme] = None
    ch_axis: Optional[int] = None
    is_dynamic: bool = False

    def __post_init__(self):
        # check dtype is one of the supported types
        # if self.dtype not in SUPPORTED_DTYPES:
        #     raise TypeError(f"Unsupported dtype {self.dtype}.")

        # check qscheme is on of the supported ones
        if self.qscheme is not None and self.qscheme not in SUPPORTED_QSCHEMES:
            raise ValueError(f"Unsupported qscheme {self.qscheme}.")

        # ch_axis must be less than the number of channels
        # but no way to check here. Just check that it is not < 0.
        if self.ch_axis is not None and self.ch_axis < 0:
            raise ValueError("Ch_axis is < 0.")

@dataclass(eq=True, frozen=True)
class FixedBFPQParamsQuantizationSpec(QuantizationSpecBase):
    dtype: Format = Format.from_shorthand("BFP[8|8]{64}(SN)")
    scale: float = 1.0
    zero_point: int = 0
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None
    qscheme: Optional[torch.qscheme] = None


"""
The way we refer to other points of quantization in the graph will be either
an input edge or an output value
input edge is the connection between input node and the node consuming the input, so it's a Tuple[Node, Node]
output value is an fx Node
"""
EdgeOrNode = Union[Tuple[Node, Node], Node]
EdgeOrNode.__module__ = "torch.ao.quantization.quantizer.quantizer"


@dataclass(eq=True, frozen=True)
class SharedBFPQuantizationSpec(BFPQuantizationSpec):
    """
    Quantization spec for the Tensors whose quantization parameters are shared with other Tensors
    """

    # the edge or node to share observer or fake quant instances with
    edge_or_node: EdgeOrNode = None


@dataclass(eq=True, frozen=True)
class DerivedBFPQuantizationSpec(BFPQuantizationSpec):
    """Quantization spec for the Tensors whose quantization parameters are derived from other Tensors"""

    derived_from: List[EdgeOrNode] = None
    derive_qparams_fn: Callable[[List[ObserverOrFakeQuantize]], Tuple[Tensor, Tensor]] = None
    dtype: Format = Format.from_shorthand("BFP[8|8]{64}(SN)")
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None
    qscheme: Optional[torch.qscheme] = None
    ch_axis: Optional[int] = None
    is_dynamic: bool = False


@dataclass
class BFPQuantizationAnnotation:
    """How are input arguemnt or output should be quantized,
    expressed as QuantizationSpec, this corresponds to how a Tensor in the
    operator Graph is observed (PTQ) or fake quantized (QAT)
    """

    # a map from torch.fx.Node to a type of QuantizationSpecBase
    input_qspec_map: Dict[Node, Optional[BFPQuantizationSpec]] = field(
        default_factory=dict
    )

    # How the output of this node is quantized, expressed as QuantizationSpec
    # TODO: change the value to QuantizationSpec in a separate PR
    output_qspec: Optional[BFPQuantizationSpec] = None

    # For a Node: node1 and edge: (node1, node2), since they are observing the same
    # Tensor, we may want to implicitly share observers, this flag allows people to
    # turn off this behavior for the output of the node
    allow_implicit_sharing: bool = True

    # whether the node is annotated or not
    _annotated: bool = False

# class BFPQuantizer(numerical.Quantize):
class BFPQuantizer(ABC):
    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        """Allows for user defined transforms to run before annotating the graph.
        This allows quantizer to allow quantizing part of the model that are otherwise not quantizable.
        For example quantizer can
        a) decompose a compound operator like scaled dot product attention,
        into bmm and softmax if quantizer knows how to quantize bmm/softmax but not sdpa
        or b) transform scalars to tensor to allow quantizing scalares.

        Note: this is an optional method
        """
        return model

    # annotate nodes in the graph with observer or fake quant constructors
    # to convey the desired way of quantization
    @abstractmethod
    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        pass

    # validate the annotated graph is supported by the backend
    @abstractmethod
    def validate(self, model: torch.fx.GraphModule) -> None:
        pass
