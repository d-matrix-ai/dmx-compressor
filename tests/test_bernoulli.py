import torch
from torch import nn


def test_sigmoid_gradients():
    """Sigmoid function itself has no gradient problems."""
    score = nn.Parameter(torch.Tensor(5, 5))
    mask = torch.sigmoid(score)
    loss = torch.sum(mask)
    loss.backward()
    print("score.grad", score.grad)
    assert score.grad is not None


def test_bernoulli_gradients():
    """Bernoulli sampler introduces gradient problem."""
    score = nn.Parameter(torch.Tensor(5, 5))
    mask = torch.sigmoid(score)
    mask = torch.bernoulli(mask)
    loss = torch.sum(mask)
    loss.backward()
    assert (score.grad is None) or (torch.sum(score.grad) == 0)
