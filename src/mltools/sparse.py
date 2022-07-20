from typing import Any

import torch
import torch.nn as nn
from parse import parse
from torch.autograd import Function

__ALL__ = [
    "Sparseness",
    "WeightSparseMixin",
    "Dense",
    "TopK",
    "BlockTopK",
    "Sparsify",
]


class Sparseness(Function):
    r"""
    This class inherits from Function, since when we use masking operations, we may have different
        forward and backward passes.
    For example, in Bernoulli, we sample the masks to 0/1 during the forward direction, while using
        the gradients of the sigmoid function in the backward pass.
    This class is to be inherited for specific masking implementations.
    Moreover, this is an abstract class for the (super)masking mechanism for network sparsification.
        Child classes to implement `get_mask()` and `from_shorthand()` method.
    """

    def __str__(self) -> str:
        raise NotImplementedError

    def get_mask(self, *input: Any):
        raise NotImplementedError

    @classmethod
    def from_shorthand(cls, sh: str):
        if sh.startswith("DENSE"):
            return Dense.from_shorthand(sh)
        elif sh.startswith("TOPK"):
            return TopK.from_shorthand(sh)
        elif sh.startswith("BTOPK"):
            return BlockTopK.from_shorthand(sh)
        elif sh.startswith("BERN"):
            return Bernoulli.from_shorthand(sh)
        else:
            raise ValueError(f"unrecognized sparseness shorthand: {sh}")


class Dense(Sparseness):
    r"""
    This is a dummy sparsity whose `forward` and `backward` are equivalent to an identity function.
    """

    def get_mask(self, score):
        return torch.ones_like(score, device=score.device)

    @classmethod
    def from_shorthand(cls, sh: str):
        return cls()

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
        assert 0 <= density <= 1.0, "density has to be between 0 and 1"
        self.density = density

    def get_mask(self, score):
        # TODO Change to autograd.Function
        _score = score.view(-1)
        idx = torch.argsort(_score, dim=0)[: int(score.numel() * (1.0 - self.density))]
        mask = (
            torch.ones_like(_score, device=score.device)
            .scatter_(dim=0, index=idx, value=0)
            .view_as(score)
        )
        return mask

    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse("TOPK{{{density:f}}}", sh)
        return cls(density=conf["density"])

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
        # TODO Change to autograd.Function
        assert (
                score.shape[self.block_dim] % self.block_size == 0
        ), f"score has size {score.shape[self.block_dim]} at dimension {self.block_dim}, not a multiple of block size {self.block_size}"
        _score = score.transpose(self.block_dim, -1)
        score_shape = _score.shape
        _score = _score.reshape(-1, self.block_size)
        idx = torch.argsort(_score, dim=1)[:, : int(self.block_size - self.K)]
        mask = (
            torch.ones_like(_score, device=score.device)
            .scatter_(dim=1, index=idx, value=0)
            .reshape(score_shape)
            .transpose_(self.block_dim, -1)
        )
        return mask

    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse("BTOPK{{{K:d}:{block_size:d},{block_dim:d}}}", sh)
        return cls(
            K=conf["K"],
            block_size=conf["block_size"],
            block_dim=conf["block_dim"],
        )

    def __str__(self) -> str:
        return f"Block TopK sparseness: pattern = {self.K}:{self.block_size}, block dimension = {self.block_dim}"

    def __repr__(self) -> str:
        return f"BTOPK{{{self.K}:{self.block_size},{self.block_dim}}}"


class Bernoulli(Sparseness):
    r"""Bernoulli sampler for supermasking."""

    @staticmethod
    def forward(ctx, score):
        sigmoid_score = torch.sigmoid(score)
        ctx.save_for_backward(sigmoid_score)
        return torch.bernoulli(sigmoid_score)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_score, = ctx.saved_tensors
        grad = sigmoid_score * (1 - sigmoid_score)
        return grad_output * grad

    def get_mask(self, score):
        return Bernoulli.apply(score)

    @classmethod
    def from_shorthand(cls, sh: str):
        return cls()

    def __str__(self) -> str:
        return f"Bernoulli sparseness"

    def __repr__(self) -> str:
        return f"BERN"


class Sparsify(nn.Module):
    r"""
    Sparsification module
    """

    def __init__(self, tensor_shape, sparseness="DENSE", backward_mode="STE"):
        """TODO Add dump_to."""
        super().__init__()
        if not isinstance(sparseness, Sparseness):
            sparseness = Sparseness.from_shorthand(sparseness)
        self.sparseness = sparseness

        # Score is set to equal to absolute value of weight (or other deterministic function).
        # Mask is a deterministic function of score.
        self.score = nn.Parameter(torch.ones(tensor_shape), requires_grad=True)
        self.score_func = lambda x: torch.abs(x)
        self.mask = torch.ones(tensor_shape)

        # Configures the backward patterns according to the mode chosen
        self.backward_mode = backward_mode
        self.enable_weight_gradient = backward_mode.lower() in {"ste", "joint"}
        self.enable_mask_gradient = backward_mode.lower() in {"supermask", "joint"}

    def forward(self, x):
        if not isinstance(self.sparseness, Dense):
            if self.training:
                mask = self.sparseness.get_mask(self.score)
                x = x if self.enable_weight_gradient else x.detach()
                mask = mask if self.enable_mask_gradient else mask.detach()
            x = x * mask
        return x

    def extra_repr(self):
        return f"sparseness = {self.sparseness.__repr__()}, backward_mode = {self.backward_mode}"


class WeightSparseMixin:
    """Mixin for weight-sparse modules."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.init_sparsifier()

    def init_sparsifier(self):
        if isinstance(
                self,
                (
                        nn.Linear,
                        nn.Bilinear,
                        nn.Embedding,
                        nn.EmbeddingBag,
                        nn.modules.conv._ConvNd,
                ),
        ):
            self.weight_sparsifier = Sparsify(self.weight.shape)
        else:
            self.weight_sparsifier = None

    def configure_sparsifier(self, sparseness, backward_mode, score_func):
        r"""
        Configures the specific parameters of the sparsifier, such as sparseness and backward method.
        If this function is not called, the sparsifier will default to "dense".
        """
        if self.weight_sparsifier is None: return
        self.weight_sparsifier.set_sparseness(sparseness)
        self.weight_sparsifier.set_score_func(score_func)
        if backward_mode == "STE":
            # TODO Check this
            self.weight_sparsifier.score.requires_grad = False
        elif backward_mode == "supermask":
            self.weight_sparsifier.score.requires_grad = True
            self.weight.requires_grad = False
            self.bias.requires_grad = False
        elif backward_mode == "joint":
            # TODO Verify other backward modes
            self.weight_sparsifier.score.requires_grad = True
            self.weight.requires_grad = True
            self.bias.requires_grad = True

    @property
    def effective_weight(self):
        return (
            self.weight
            if self.weight_sparsifier is None
            else self.weight_sparsifier(self.weight)
        )

    @property
    def weight_sparseness(self):
        return (
            repr(self.weight_sparsifier.sparseness)
            if self.weight_sparsifier is not None
            else None
        )
