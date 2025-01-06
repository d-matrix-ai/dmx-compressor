import torch

import torch.nn.functional as F
from dmx.compressor.modeling.nn import ScaledDotProductAttention
from transformers import pipeline
from dmx.compressor.modeling import DmxModel


torch.manual_seed(10)
query = torch.rand(100, 1024)
key = torch.rand(100, 1024)
value = torch.rand(100, 1024)
attn_mask = torch.zeros(100, 100)
scale = 5.0


def test_no_kwargs():
    mod = ScaledDotProductAttention()
    gm = mod.module_graph(query, key, value)
    out1 = gm(query, key, value)
    out0 = F.scaled_dot_product_attention(query, key, value)
    assert torch.all(out1 - out0 < 1e-5), "Mismatch at no kwargs"


def test_attn_mask():
    mod = ScaledDotProductAttention()
    gm = mod.module_graph(query, key, value, attn_mask=attn_mask)
    out1 = gm(query, key, value, attn_mask=attn_mask)
    out0 = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
    assert torch.all(out1 - out0 < 1e-5), "Mismatch at attn_mask"


def test_dropout():
    mod = ScaledDotProductAttention(dropout_p=1)
    gm = mod.module_graph(query, key, value)
    out1 = gm(query, key, value)
    out0 = F.scaled_dot_product_attention(query, key, value, dropout_p=1)
    assert torch.all(out1 - out0 < 1e-5), "Mismatch at dropout"


def test_causal():
    mod = ScaledDotProductAttention()
    gm = mod.module_graph(query, key, value, is_causal=True)
    out1 = gm(query, key, value)
    out0 = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    assert torch.all(out1 - out0 < 1e-5), "Mismatch at is_causal"


def test_scale():
    mod = ScaledDotProductAttention()
    gm = mod.module_graph(query, key, value, scale=scale)
    out1 = gm(query, key, value, scale)
    out0 = F.scaled_dot_product_attention(query, key, value, scale=scale)
    assert torch.all(out1 - out0 < 1e-4), "Mismatch at scale"


def test_gqa():
    query = torch.rand(1, 8, 100, 1024)
    key = torch.rand(1, 2, 100, 1024)
    value = torch.rand(1, 2, 100, 1024)
    mod = ScaledDotProductAttention()
    gm = mod.module_graph(query, key, value, enable_gqa=True)
    out1 = gm(query, key, value)
    out0 = F.scaled_dot_product_attention(query, key, value, enable_gqa=True)
    assert torch.all(out1 - out0 < 1e-5), "Mismatch at enable_gqa"


def test_distilgpt2():
    pipe = pipeline(
        task="text-generation",
        model="d-matrix/distilgpt2",
        trust_remote_code=True,
        device="cpu",
    )
    pipe.model.eval()
    x = torch.randint(1, 100, (1, 1024))
    with torch.no_grad():
        y1 = pipe.model(x, labels=x)
    pipe.model = DmxModel.from_torch(pipe.model)
    with torch.no_grad():
        y2 = pipe.model(x, labels=x)
    assert y1.loss.item() - y2.loss.item() < 1e-5, "Mismatch in distilgpt2 loss"
