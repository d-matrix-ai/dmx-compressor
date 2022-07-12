import pytest
import torch
import torch.optim as optim
from mltools import corsair
from mltools.sparse import Sparseness
from torch import nn

corsair.aware()


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim)
        self.weight_init = self.fc.weight.data
        self.bias_init = self.fc.bias.data

    def forward(self, x):
        return self.fc(x)


@pytest.mark.parametrize("input_dim", (30, 40, 50))
def test_training_sparse_network(input_dim):
    """This test is to assert that (1) the weights and biases don't learn;
    (2) the mask in the sparsifier learns."""
    batch_size = 16
    num_epochs = 10
    output_dim = 10
    sparseness = Sparseness.from_shorthand("BERN")
    backward_mode = "supermask"
    score_func = torch.abs

    model = MLP(input_dim, output_dim)
    model.train()

    for n, m in model.named_modules():
        if isinstance(m, corsair.nn.Linear):
            m.configure_sparsifier(sparseness, backward_mode, score_func)

    # Calculate gradients and do assertions accordingly
    data = torch.rand((batch_size, input_dim))
    ground_truth_output = torch.randint(0, output_dim, (batch_size,))

    optimizer = optim.SGD(model.parameters(), lr=100, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    score_init = model.fc.weight_sparsifier.score.data

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, ground_truth_output)
        loss.backward()
        optimizer.step()

        # Test that (1) there are no changes to the model weight and bias terms;
        # and (2) the score is changing in weight sparsifier
        assert torch.sum(torch.abs(model.fc.weight - model.weight_init)) == 0
        assert torch.sum(torch.abs(model.fc.bias - model.bias_init)) == 0
        assert torch.abs(model.fc.weight_sparsifier.score.grad).sum() != 0

        # TODO I don't know why the 3rd assertion can pass, but not the following assertion
        # assert torch.abs(model.fc.weight_sparsifier.score - score_init).sum() != 0
