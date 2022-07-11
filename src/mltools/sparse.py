from typing import Any

import torch
import torch.nn as nn
from parse import parse

__ALL__ = [
    "Sparseness",
    "WeightSparseMixin",
    "Dense",
    "TopK",
    "BlockTopK",
    "Sparsify",
]


class Sparseness:
    r"""
    This is an abstract class of tensor sparseness.
    Child classes to implement `get_mask()` and `from_shorthand()` method.
    """

    def __init__(self):
        pass

    def __str__(self) -> str:
        raise NotImplementedError

    def get_mask(self, *input: Any):
        raise NotImplementedError

    @staticmethod
    def from_shorthand(sh: str):
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
    This is a dummy sparsity whose `get_mask()` returns ones.
    """

    def __init__(self):
        super().__init__()

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
    r"""
    Bernoulli sampler for supermasking
    """

    def __init__(self):
        super().__init__()

    def get_mask(self, score):
        # return torch.sign(x) * torch.bernoulli(torch.abs(x))
        return torch.sigmoid(score)
        return torch.bernoulli(score)

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

    def __init__(self, tensor_shape, sparseness="DENSE", backward_mode="STE", dump_to=None):
        super().__init__()
        self.set_sparseness(sparseness)

        # Score is set to equal to absolute value of weight (or other deterministic function).
        # Mask is a deterministic function of score.
        self.score = nn.Parameter(torch.Tensor(tensor_shape), requires_grad=True)
        self.score_initialized = False
        self.score_func = torch.abs
        self.mask = torch.ones(tensor_shape)

        self.dump_to = dump_to
        self.backward_mode = backward_mode

    def set_sparseness(self, sparseness="DENSE"):
        if not isinstance(sparseness, Sparseness):
            sparseness = Sparseness.from_shorthand(sparseness)
        self.sparseness = sparseness

    def set_score_func(self, score_func):
        self.score_func = score_func

    def set_score(self, x=None):
        # Only sets the scores once; we don't allow joint training of score and mask for now.
        if self.score_initialized: return
        if self.score_func is None: return
        with torch.no_grad():
            score = self.score_func(x)
            self.score.copy_(score)
            self.score_initialized = True

    def forward(self, x):
        if not isinstance(self.sparseness, Dense):
            if not self.score_initialized:
                self.set_score(x)
            if self.training:
                self.mask = self.sparseness.get_mask(self.score)
            x = x * self.mask
        if self.dump_to is not None:
            # TODO
            pass
        return x

    def extra_repr(self):
        return f"sparseness = {self.sparseness.__repr__()}, backward_mode = {self.backward_mode}"


class WeightSparseMixin:
    """
    Mixin for weight-sparse modules
    """

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
        """
        Configures the specific parameters of the sparsifier, such as sparseness and backward method.
        If this function is not called, the sparsifier will default to "dense".
        """
        if self.weight_sparsifier is None: return
        self.weight_sparsifier.set_sparseness(sparseness)
        self.weight_sparsifier.set_score_func(score_func)
        if backward_mode == "STE":
            # TODO Check this
            self.weight_sparsifier.eval()
        elif backward_mode == "supermask":
            self.weight.requires_grad = False
            self.bias.requires_grad = False
        else:
            # TODO Verify other backward modes
            pass

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
