from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from torch.library import custom_op
import torch.nn as nn
import transformers
from packaging import version

schema = "(Tensor q, Tensor k, Tensor cos, Tensor sin, Tensor? position_ids=None, int unsqueeze_dim=1) -> (Tensor, Tensor)"
apply_rotary_pos_emb = custom_op(
    "dmx_ops::apply_rotary_pos_emb",
    apply_rotary_pos_emb,
    mutates_args=(),
    schema=schema,
)


@apply_rotary_pos_emb.register_fake
def _(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    return q, k


if version.parse(transformers.__version__) >= version.parse("4.52.4"):
    transformers.models.llama.modeling_llama.apply_rotary_pos_emb = apply_rotary_pos_emb
    transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb = apply_rotary_pos_emb
