from collections import OrderedDict
import torch
from torch import Tensor, Size
from torch.fx import Graph, symbolic_trace
import transformers
import transformers.activations

from dmx.compressor.numerical import Same, CastTo, CastToDict
from . import DmxModule
from .torch_modules import GELUBase
from .core import DmxGraph


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

    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        g = DmxGraph()
        with g.inserting_after():
            _input_dq = g.create_placeholders(
                ["_input"],
                ["input_casts.input_cast"],
                [repr(self.input_casts.input_cast.format)],
            )
            _weight_dq = g.get_attr(
                "weight", "weight_cast", repr(self.weight_cast.format)
            )

            # Non Tensor Attributes (no need to quantize)
            eps = g.get_attr("eps")

            args = ((_input_dq), _weight_dq, eps)
            _output_dq = g.create_node(
                "call_function",
                self.functional_forward,
                args,
                name="GemmanRMSNorm",
                cast_name="output_casts.output_cast",
                cast_format=repr(self.output_casts.output_cast.format),
            )
            g.output(_output_dq)
        return g


class NewGELU(GELUBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(transformers.activations.NewGELUActivation, *args, **kwargs)

    def functional_forward(self, x):
        return transformers.activations.NewGELUActivation().forward(x)


class FastGELU(GELUBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(transformers.activations.FastGELUActivation, *args, **kwargs)

    def functional_forward(self, x):
        return transformers.activations.FastGELUActivation().forward(x)


class QuickGELU(GELUBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(transformers.activations.QuickGELUActivation, *args, **kwargs)

    def functional_forward(self, x):
        return transformers.activations.QuickGELUActivation().forward(x)


class ClippedGELU(GELUBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            transformers.activations.ClippedGELUActivation, *args, **kwargs
        )

    def functional_forward(self, x):
        return transformers.activations.ClippedGELUActivation(
            self.min, self.max
        ).forward(x)


class BloomGELU(GELUBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            transformers.models.bloom.modeling_bloom.BloomGelu, *args, **kwargs
        )

    def functional_forward(self, x):
        return transformers.models.bloom.modeling_bloom.BloomGelu().forward(x)


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
        g = DmxGraph()
        with g.inserting_after():
            # a function that insert placeholder nodes
            dq_q, dq_k, dq_cos, dq_sin = g.create_placeholders(
                ["q", "k", "cos", "sin"],
                [
                    "input_casts.q_cast",
                    "input_casts.k_cast",
                    "input_casts.cos_cast",
                    "input_casts.sin_cast",
                ],
                [
                    repr(self.input_casts.q_cast.format),
                    repr(self.input_casts.k_cast.format),
                    repr(self.input_casts.cos_cast.format),
                    repr(self.input_casts.sin_cast.format),
                ],
            )
            q_res = g.call_function(
                apply_rotary_embeddings,
                (dq_q, dq_cos, dq_sin),
                cast_name="output_casts.q_embed_cast",
                cast_format=repr(self.output_casts.q_embed_cast.format),
            )
            k_res = g.call_function(
                apply_rotary_embeddings,
                (dq_k, dq_cos, dq_sin),
                cast_name="output_casts.k_embed_cast",
                cast_format=repr(self.output_casts.k_embed_cast.format),
            )

            g.output((q_res, k_res))
        return g


def apply_rotary_embeddings(x, cos_embedding, sin_embedding):
    cos = cos_embedding.unsqueeze(1)
    sin = sin_embedding.unsqueeze(1)
    x_embed = (x * cos) + (ApplyRotaryPosEmbBase().rotate_half(x) * sin)
    return x_embed


class LlamaRotaryEmbedding(
    DmxModule, transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
):
    def __init__(self, config, device=None):
        super().__init__(config, device)
        self.cos = None
        self.sin = None
        self.input_casts = CastToDict(
            OrderedDict(
                {
                    "x_cast": CastTo(),
                    "position_ids_cast": CastTo(),
                }
            )
        )
        self.output_casts = CastToDict(
            OrderedDict({"cos_cast": CastTo(), "sin_cast": CastTo()})
        )

    def _forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        _output = self.approx_forward((x, cos, sin))
        return _output

    def _forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        self.sin = sin.to(dtype=x.dtype)
        self.cos = cos.to(dtype=x.dtype)
        return self.cos, self.sin

    @classmethod
    def from_raw(cls, raw: torch.nn.Module) -> DmxModule:
        initial_dmx = cls(raw.config, raw.inv_freq.device)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx

    def to_compiler_graph(self) -> Graph:
        if self.sin is None:
            raise ValueError(
                "forward of LlamaROPE must be called before to_compiler_graph"
            )

        g = DmxGraph()
        with g.inserting_after():
            g.create_placeholders(["x", "position_ids"])
            sin = g.get_attr("sin")
            cos = g.get_attr("cos")
            g.output((cos, sin))
        return g
