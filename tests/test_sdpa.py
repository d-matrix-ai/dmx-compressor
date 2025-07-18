import torch

import torch.nn.functional as F
from dmx.compressor.modeling.nn import ScaledDotProductAttention
from dmx.compressor.modeling import DmxModel


torch.manual_seed(0)
query = torch.rand(2, 100, 1024)
key = torch.rand(2, 100, 1024)
value = torch.rand(2, 100, 1024)
attn_mask = torch.ones(100, 100)
scale = 5


def test_no_kwargs():
    mod = ScaledDotProductAttention()
    gm = mod.module_graph(query, key, value)
    out1 = gm(query, key, value)
    out0 = F.scaled_dot_product_attention(query, key, value)
    assert torch.allclose(out1, out0), "Mismatch at no kwargs"


def test_attn_mask():
    mod = ScaledDotProductAttention()
    gm = mod.module_graph(query, key, value, attn_mask=attn_mask)
    out1 = gm(query, key, value, attn_mask=attn_mask)
    out0 = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
    assert torch.allclose(out1, out0), "Mismatch at attn_mask"


def test_causal():
    mod = ScaledDotProductAttention()
    gm = mod.module_graph(query, key, value, is_causal=True)
    out1 = gm(query, key, value)
    out0 = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    assert torch.allclose(out1, out0), "Mismatch at is_causal"


def test_scale():
    mod = ScaledDotProductAttention()
    gm = mod.module_graph(query, key, value, scale=scale)
    out1 = gm(query, key, value, scale)
    out0 = F.scaled_dot_product_attention(query, key, value, scale=scale)
    assert torch.allclose(out1, out0, atol=1e-4), "Mismatch at scale"


def test_gqa():
    query = torch.rand(1, 8, 100, 1024)
    key = torch.rand(1, 2, 100, 1024)
    value = torch.rand(1, 2, 100, 1024)
    mod = ScaledDotProductAttention()
    gm = mod.module_graph(query, key, value, enable_gqa=True)
    out1 = gm(query, key, value)
    out0 = F.scaled_dot_product_attention(query, key, value, enable_gqa=True)
    assert torch.allclose(out1, out0), "Mismatch at enable_gqa"


class CustomMod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(512, 512)
        self.value = torch.nn.Linear(512, 512)
        self.query = torch.nn.Linear(512, 512)

    def forward(self, x, s):
        key = self.key(x)
        value = self.value(x)
        query = self.query(x)
        out = F.scaled_dot_product_attention(query, key, value, scale=s)
        return out


class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mods = torch.nn.ModuleList([CustomMod() for _ in range(5)])

    def forward(self, x, s=None):
        for mod in self.mods:
            x = mod(x, s)
        return x


def test_larger_model():
    model = CustomModel()
    x = torch.rand(1, 512)
    with torch.no_grad():
        y1 = model(x)
    model = DmxModel.from_torch(model)
    with torch.no_grad():
        y2 = model(x)
    assert torch.allclose(y1, y2), "Mismatch at custom model"
