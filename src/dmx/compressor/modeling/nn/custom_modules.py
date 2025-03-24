import math
from typing import Union, List, Optional
from collections import OrderedDict
import torch
from torch import Tensor, Size
import torch.nn.functional as F
from torch.fx import Graph, symbolic_trace
import transformers
import transformers.activations

from dmx.compressor.numerical import Same, CastTo, CastToDict
from . import DmxModule
from .torch_modules import GELUBase


class GemmaRMSNorm(DmxModule, transformers.models.gemma.modeling_gemma.GemmaRMSNorm):
    r"""
    An extension of RMSNorm layer to support DmxModule configurations.
    This module performs RMS-based layer normalization on the input tensor.
    The layer normalization is characterized by the `hidden_size` and an optional `eps` value for numerical stability.

    Args:
        dim (int): The size of the hidden layer (number of hidden units).
        eps (float, optional): A small constant added to the denominator for numerical stability. Defaults to 1e-6.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the RMS layer normalization.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(dim, eps=eps)

    def _norm(self, x, eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def functional_forward(self, x, weight, eps):
        output = self._norm(x.float(), eps)
        output = output * (1.0 + weight.float())
        return output.type_as(x)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,), self._weight, self.eps)
        return _output

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new RMSNorm object (DmxModule) from a given PyTorch RMSNorm layer.

        Args:
            raw (torch.nn.Module): A PyTorch RMSNorm layer to be converted.

        Returns:
            DmxModule: A RMSNorm object that has the same configuration as the input PyTorch RMSNorm layer.
        """
        initial_dmx = cls(
            dim=raw.weight.shape[0],
            eps=raw.variance_epsilon if hasattr(raw, "variance_epsilon") else raw.eps,
        )
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class NewGELU(GELUBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(transformers.activations.NewGELUActivation, *args, **kwargs)


class FastGELU(GELUBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(transformers.activations.FastGELUActivation, *args, **kwargs)


class QuickGELU(GELUBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(transformers.activations.QuickGELUActivation, *args, **kwargs)


class ClippedGELU(GELUBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            transformers.activations.ClippedGELUActivation, *args, **kwargs
        )


class BloomGELU(GELUBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            transformers.models.bloom.modeling_bloom.BloomGelu, *args, **kwargs
        )


class ApplyRotaryPosEmbBase(torch.nn.Module):
    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, cos, sin, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """

        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


class ApplyRotaryPosEmb(DmxModule, ApplyRotaryPosEmbBase):
    def __init__(self) -> None:
        super().__init__()
        self.input_casts = CastToDict(
            OrderedDict(
                {
                    "q_cast": CastTo(),
                    "k_cast": CastTo(),
                    "cos_cast": CastTo(),
                    "sin_cast": CastTo(),
                }
            )
        )
        self.output_casts = CastToDict(
            OrderedDict({"q_embed_cast": CastTo(), "k_embed_cast": CastTo()})
        )

    def _forward(self, q, k, cos, sin, unsqueeze_dim=1) -> Tensor:
        _output = self.approx_forward((q, k, cos, sin, unsqueeze_dim))
        return _output

    def to_compiler_graph(self):
        # combine these two lines into one context manager
        g = Graph()
        with g.inserting_after():
            # a function that insert placeholder nodes
            placeholders = self.create_placeholders(g, ["q", "k", "cos", "sin"])
            cast_names = [
                "input_casts.q_cast",
                "input_casts.k_cast",
                "input_casts.cos_cast",
                "input_casts.sin_cast",
            ]
            dq_q, dq_k, dq_cos, dq_sin = self.qdq_nodes(g, placeholders, cast_names)
            q_emb = g.call_function(apply_rotary_embeddings, (dq_q, dq_cos, dq_sin))
            k_emb = g.call_function(apply_rotary_embeddings, (dq_k, dq_cos, dq_sin))
            output_cast_names = [
                "output_casts.q_embed_cast",
                "output_casts.k_embed_cast",
            ]
            q_res, k_res = self.qdq_nodes(g, [q_emb, k_emb], output_cast_names)
            # output = g.call_function(tuple, (q_res, k_res))
            g.output((q_res, k_res))
        return g


def apply_rotary_embeddings(x, cos_embedding, sin_embedding):
    cos = cos_embedding.unsqueeze(1)
    sin = sin_embedding.unsqueeze(1)
    x_embed = (x * cos) + (ApplyRotaryPosEmbBase().rotate_half(x) * sin)
    return x_embed
