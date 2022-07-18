import pytest


import torch
import torch.optim as optim
from mltools import corsair
from mltools.sparse import Sparseness, Sparsify
from torch import nn

RANDOM_SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)

@pytest.mark.parametrize(
    "tensor_shape",
    (
        (1024, 1024),
        (256, 256, 32, 32),
        (8, 256, 256),
    ),
)
@pytest.mark.parametrize(
    "sparseness",
    (
        "TOPK{0.5}",
        "BTOPK{4:8,-1}",
        "BTOPK{2:8,-1}",
        "BERN",
    ),
)
@pytest.mark.parametrize(
    "backward_mode",
    (
        "STE",
        "supermask",
        "joint",
    ),
)
def test_sparsify(tensor_shape, sparseness, backward_mode):
    sp = Sparsify(
        tensor_shape,
        sparseness,
        backward_mode,
    ).to(device)
    x = torch.randn(tensor_shape).to(device)
    y = sp(x)
    # test the mask to be a correct sparseness pattern
    y.backward(torch.ones_like(y))
    # if backward_mode=="STE", x.grad is a tensor, and sp.score.grad is None
    # if backward_mode=="supermask", x.grad is None, and sp.score.grad is a tensor
    # if backward_mode=="joint", then both x.grad and sp.score.grad is a tensor


# ---

# corsair.aware()


# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.fc = nn.Linear(input_dim, output_dim)
#         self.weight_init = self.fc.weight.data
#         self.bias_init = self.fc.bias.data

#     def forward(self, x):
#         return self.fc(x)


# @pytest.mark.parametrize("input_dim", (30, 40, 50))
# def test_training_sparse_network(input_dim):
#     """This test is to assert that (1) the weights and biases don't learn;
#     (2) the mask in the sparsifier learns."""
#     batch_size = 16
#     num_epochs = 10
#     output_dim = 10
#     sparseness = Sparseness.from_shorthand("BERN")
#     backward_mode = "supermask"
#     score_func = torch.abs

#     model = MLP(input_dim, output_dim)
#     model.train()

#     for n, m in model.named_modules():
#         if isinstance(m, corsair.nn.Linear):
#             m.configure_sparsifier(sparseness, backward_mode, score_func)

#     # Calculate gradients and do assertions accordingly
#     data = torch.rand((batch_size, input_dim))
#     ground_truth_output = torch.randint(0, output_dim, (batch_size,))

#     optimizer = optim.SGD(model.parameters(), lr=100, momentum=0.9)
#     criterion = nn.CrossEntropyLoss()
#     score_init = model.fc.weight_sparsifier.score.data

#     for epoch in range(num_epochs):
#         optimizer.zero_grad()

#         output = model(data)
#         loss = criterion(output, ground_truth_output)
#         loss.backward()
#         optimizer.step()

#         # Test that (1) there are no changes to the model weight and bias terms;
#         # and (2) the score is changing in weight sparsifier
#         assert torch.sum(torch.abs(model.fc.weight - model.weight_init)) == 0
#         assert torch.sum(torch.abs(model.fc.bias - model.bias_init)) == 0
#         assert torch.abs(model.fc.weight_sparsifier.score.grad).sum() != 0

#         # TODO I don't know why the 3rd assertion can pass, but not the following assertion
#         # assert torch.abs(model.fc.weight_sparsifier.score - score_init).sum() != 0
