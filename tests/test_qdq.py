import torch
from dmx.compressor import nn
import dmx.ops
import pytest

torch.manual_seed(0)


def test_linear():
    linear = nn.Linear(64, 64)
    gm = torch.fx.GraphModule(linear, linear.to_compiler_graph())
    inp = torch.rand((1, 64))
    out0 = linear(inp)
    out1 = gm(inp)
    assert torch.allclose(out0, out1, atol=1e-5)


def test_resadd():
    resadd = nn.ResAdd()
    gm = torch.fx.GraphModule(resadd, resadd.to_compiler_graph())
    inp = torch.rand((1, 64))
    out0 = resadd(inp, inp)
    out1 = gm(inp, inp)
    assert torch.allclose(out0, out1, atol=1e-5)


def test_mul():
    mul = nn.Mul()
    gm = torch.fx.GraphModule(mul, mul.to_compiler_graph())
    inp = torch.rand((1, 64))
    out0 = mul(inp, inp)
    out1 = gm(inp, inp)
    assert torch.allclose(out0, out1, atol=1e-5)


def test_actactmatmul():
    actactmatmul = nn.ActActMatMul()
    gm = torch.fx.GraphModule(actactmatmul, actactmatmul.to_compiler_graph())
    inp1 = torch.rand((1, 64))
    inp2 = torch.rand((64, 4))
    out0 = actactmatmul(inp1, inp2)
    out1 = gm(inp1, inp2)
    assert torch.allclose(out0, out1, atol=1e-5)


def test_baddbmm():
    baddbmm = nn.BAddBMM()
    gm = torch.fx.GraphModule(baddbmm, baddbmm.to_compiler_graph())
    inp1 = torch.rand((10, 3, 4))
    inp2 = torch.rand((10, 3, 64))
    inp3 = torch.rand((10, 64, 4))
    out0 = baddbmm(inp1, inp2, inp3)
    out1 = gm(inp1, inp2, inp3)
    assert torch.allclose(out0, out1, atol=1e-5)


def test_embed():
    embed = nn.Embedding(10, 64)
    gm = torch.fx.GraphModule(embed, embed.to_compiler_graph())
    inp = torch.randint(0, 10, (1, 10))
    out0 = embed(inp)
    out1 = gm(inp)
    assert torch.allclose(out0, out1, atol=1e-5)


@pytest.mark.parametrize(
    "module_type",
    (
        "RMSNorm",
        "LayerNorm",
        "GemmaRMSNorm",
    ),
)
def test_norms(module_type):
    norm = eval(f"nn.{module_type}(64)")
    gm = torch.fx.GraphModule(norm, norm.to_compiler_graph())
    inp = torch.rand((1, 64))
    out0 = norm(inp)
    out1 = gm(inp)
    assert torch.allclose(out0, out1, atol=1e-5)


def test_groupnorm():
    groupnorm = nn.GroupNorm(8, 64)
    gm = torch.fx.GraphModule(groupnorm, groupnorm.to_compiler_graph())
    inp = torch.rand((1, 64))
    out0 = groupnorm(inp)
    out1 = gm(inp)
    assert torch.allclose(out0, out1, atol=1e-5)


def test_dropout():
    dropout = nn.Dropout(0)
    gm = torch.fx.GraphModule(dropout, dropout.to_compiler_graph())
    inp = torch.rand((1, 64))
    out0 = dropout(inp)
    out1 = gm(inp)
    assert torch.allclose(out0, out1, atol=1e-5)


def test_clippedgelu():
    act = nn.ClippedGELU(0, 1)
    gm = torch.fx.GraphModule(act, act.to_compiler_graph())
    inp = torch.rand((1, 64))
    out0 = act(inp)
    out1 = gm(inp)
    assert torch.allclose(out0, out1, atol=1e-5)


@pytest.mark.parametrize(
    "module_type",
    (
        "Softmax",
        "ReLU",
        "ReLU6",
        "SiLU",
        "Tanh",
        "NewGELU",
        "FastGELU",
        "QuickGELU",
        "BloomGELU",
    ),
)
def test_activations(module_type):
    act = eval(f"nn.{module_type}()")
    gm = torch.fx.GraphModule(act, act.to_compiler_graph())
    inp = torch.rand((1, 64))
    out0 = act(inp)
    out1 = gm(inp)
    assert torch.allclose(out0, out1, atol=1e-5)


def test_apply_rope():
    rope = nn.ApplyRotaryPosEmb()
    q = torch.randn(1, 64, 64)
    k = torch.randn(1, 64, 64)
    cos = torch.randn(1, 64)
    sin = torch.randn(1, 64)

    out0 = rope(q, k, cos, sin)
    qdq = rope.to_compiler_graph()
    qdq_module = torch.fx.GraphModule(rope, qdq)
    out1 = qdq_module(q, k, cos, sin)
    assert torch.allclose(out0[0], out1[0], atol=1e-5)
    assert torch.allclose(out0[1], out1[1], atol=1e-5)
