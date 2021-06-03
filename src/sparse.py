import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


__ALL__ = [
    "WeightSparseMixin",
    "Dense",
    "TopK",
    "BlockTopK",
    "Sparsify",
]


class Sparseness:
    r"""
    This is an abstract class of tensor sparseness.
    Child classes to implement `get_mask()` method.
    """

    def __init__(self):
        pass

    def __str__(self) -> str:
        raise NotImplementedError

    def get_mask(self, *input: Any) -> None:
        raise NotImplementedError


class Dense(Sparseness):
    r"""
    This is a dummy sparsity whose `get_mask()` returns ones.
    """

    def __init__(self):
        super().__init__()

    def get_mask(self, score):
        return torch.ones_like(score, device=score.device)

    def __str__(self) -> str:
        return f"Dummy sparseness: no pruning"

    def __repr__(self) -> str:
        return f"DENSE"


class TopK(Sparseness):
    r"""
    Fine-grain unstructured sparsity with top-K scored entries non-zero
    """

    def __init__(self, density=0.5):
        super().__init__()
        self.density = density

    def get_mask(self, score):
        idx = score.view(-1).argsort()
        mask = torch.ones_like(idx, device=score.device)
        mask[idx[: math.round(score.numel() * (1.0 - self.density))]] = 0
        return mask.view_as(score)

    def __str__(self) -> str:
        return f"Global TopK sparseness: density = {self.density}"

    def __repr__(self) -> str:
        return f"TOPK{{{self.density}}}"


class BlockTopK(Sparseness):
    r"""
    Fine-grain structured sparsity with K non-zeros out of `block_size` elements along `block_dim`.
    """

    def __init__(self, K=4, block_size=8, block_dim=-1):
        super().__init__()
        assert 0 < K <= block_size, "N and M must be positive and N no greater than M"
        self.K = K
        self.block_size = block_size
        self.block_dim = block_dim

    def get_mask(self, score):
        assert (
            score.shape[self.block_dim] % self.block_size == 0
        ), f"score has size {score.shape[self.block_dim]} at dimension {self.block_dim}, not a multiple of block size {self.block_size}"
        _score = score.transpose(self.block_dim, -1)
        score_shape = _score.shape
        idx = torch.argsort(_score.reshape(-1, self.block_size), dim=1)[
            :, : int(self.block_size - self.K)
        ]
        mask = (
            torch.ones_like(_score, device=score.device)
            .scatter_(dim=1, index=idx, value=0)
            .reshape(score_shape)
            .transpose_(self.block_dim, -1)
        )
        return mask

    def __str__(self) -> str:
        return f"Block TopK sparseness: pattern = {self.K}:{self.block_size}, block dimension = {self.block_dim}"

    def __repr__(self) -> str:
        return f"BTOPK{{{self.K}:{self.block_size},{self.block_dim}}}"


class Bernoulli(Sparseness):
    r"""
    Bernoulli sampler for supermasking
    """

    def __init__(self):
        super().__init__()

    def get_mask(self, score):
        # return torch.sign(x) * torch.bernoulli(torch.abs(x))
        return torch.bernoulli(score)

    def __str__(self) -> str:
        return f"Bernoulli sparseness"

    def __repr__(self) -> str:
        return f"BERN"


class Sparsifier(Function):
    r"""
    Sparsifier class
    """

    @staticmethod
    def do_backward(x, score, mask, g_x, g_score, mode):
        # TODO: refactor this
        if mode == "STE":
            return g_x, None
        elif mode == "supermask":
            return None, g_score
        elif mode == "joint":
            return g_x, g_score
        elif mode == "NM":
            return (
                g_x + 2e-4 * (1 - mask) * x,
                None,
            )  # https://github.com/NM-sparsity/NM-sparsity
        else:
            raise ValueError(f"unsupported backward mode: {mode}")

    @staticmethod
    def forward(ctx, x, score, sp, mode="STE"):
        ctx.mode = mode
        mask = sp.get_mask(score)
        ctx.save_for_backward(x, score, mask)
        return x * mask

    @staticmethod
    def backward(ctx, g_x, g_score):
        x, score, mask = ctx.saved_variables
        _g_x, _g_score = Sparsifier.do_backward(x, score, mask, g_x, g_score)
        return _g_x, _g_score, None, None


class Sparsify(nn.Module):
    r"""
    Sparsification module
    """

    def __init__(
        self, tensor_shape, sparseness=Dense(), backward_mode="STE", dump_to=None
    ):
        super().__init__()
        self.score = nn.Parameter(torch.Tensor(tensor_shape))
        self.sparseness = sparseness
        self.backward_mode = backward_mode
        self.dump_to = dump_to

    def set_score(self, score_value):
        assert (
            score_value.shape == self.score.shape
        ), "setting score has to be in the same shape as the weight"
        self.score.data = score_value

    def forward(self, x):
        assert x.shape == self.score.shape, "score and x have to be of the same shape"
        if not isinstance(self.sparseness, Dense):
            x = Sparsifier.apply(x, self.score, self.sparseness, self.backward_mode)
        self.mask = self.sparseness.get_mask(self.score)
        if self.dump_to is not None:
            pass
        return x

    def extra_repr(self):
        return f"sparseness = {self.sparseness.__repr__()}, backward_mode = {self.backward_mode}"


class WeightSparseMixin:
    """
    Mixin for weight-sparse modules
    """

    def init_sparsifier(self):
        if (
            type(self)
            in (
                nn.Linear,
                nn.Bilinear,
                nn.Embedding,
                nn.EmbeddingBag,
            )
            or isinstance(self, nn.modules.conv._ConvNd)
        ):
            self.weight_sparsifier = Sparsify(self.weight.shape)
            self.weight_sparsifier.set_score(torch.abs(self.weight))
        else:
            self.weight_sparsifier = None

    @property
    def effective_weight(self):
        return self.weight_sparsifier(self.weight)


if __name__ == "__main__":
    pass
