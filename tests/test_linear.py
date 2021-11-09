import pytest
import torch
import torch.nn.functional as F
from mltools import corsair


RANDOM_SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)


@pytest.mark.parametrize("bias", (True, False))
@pytest.mark.parametrize("in_features,out_features", ((8, 8), (64, 64), (1024, 1024)))
@pytest.mark.parametrize("leading_dims", ((1,), (64,), (1, 16), (64, 16)))
def test_linear(leading_dims, in_features, out_features, bias):
    torch_module = torch.nn.Linear(in_features, out_features, bias, device=device)
    corsair_module = corsair.nn.Linear(in_features, out_features, bias, device=device)
    corsair_module.weight.data = torch_module.weight.data
    if bias:
        corsair_module.bias.data = torch_module.bias.data
    t_inp = torch.randn(*leading_dims, in_features, device=device).requires_grad_()
    c_inp = t_inp.clone().detach().requires_grad_()
    t_out = torch_module(t_inp)
    c_out = corsair_module(c_inp)
    g_out = torch.randn_like(t_out)
    t_out.backward(g_out)
    c_out.backward(g_out)
    assert torch.allclose(t_out.data, c_out.data, atol=1e-6)
    assert torch.allclose(t_inp.grad, c_inp.grad, atol=1e-6)
