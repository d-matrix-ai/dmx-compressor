import pytest
import torch
from dmx.compressor import dmx


RANDOM_SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)


@pytest.mark.parametrize(
    "num_embeddings,embedding_dim",
    (
        (8, 16),
        (16384, 64),
    ),
)
@pytest.mark.parametrize(
    "batch_size",
    (
        1,
        4,
    ),
)
def test_conv2d(
    batch_size,
    num_embeddings,
    embedding_dim,
):
    torch_module = torch.nn.Embedding(num_embeddings, embedding_dim, device=device)
    dmx_module = dmx.nn.Embedding(num_embeddings, embedding_dim, device=device)
    dmx_module.weight.data = torch_module.weight.data
    t_inp = torch.randint(0, num_embeddings, (batch_size,), device=device)
    c_inp = t_inp.clone().detach()
    t_out = torch_module(t_inp)
    c_out = dmx_module(c_inp)
    assert torch.allclose(t_out.data, c_out.data)
    assert torch.allclose(torch_module.weight.data, dmx_module.weight.data)
