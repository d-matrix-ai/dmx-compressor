from typing import Any
from parse import parse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


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
        return torch.bernoulli(score)

    @classmethod
    def from_shorthand(cls, sh: str):
        return cls()

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
        ctx.set_materialize_grads(False)
        ctx.mode = mode
        mask = sp.get_mask(score)
        ctx.save_for_backward(x, score, mask)
        return x * mask, score

    @staticmethod
    def backward(ctx, g_x, g_score):
        x, score, mask = ctx.saved_variables
        _g_x, _g_score = Sparsifier.do_backward(x, score, mask, g_x, g_score, ctx.mode)
        return _g_x, _g_score, None, None


class Sparsify(nn.Module):
    r"""
    Sparsification module
    """

    def __init__(
        self, tensor_shape, sparseness="DENSE", backward_mode="STE", dump_to=None
    ):
        super().__init__()
        self.score = nn.Parameter(torch.Tensor(tensor_shape))
        if not isinstance(sparseness, Sparseness):
            sparseness = Sparseness.from_shorthand(sparseness)
        self.sparseness = sparseness
        self.backward_mode = backward_mode
        self.dump_to = dump_to
        self.mask = torch.ones(tensor_shape)
        self.score_func = None  # torch.abs

    def set_score(self, x):
        if self.score_func is not None:
            with torch.no_grad():
                score_value = self.score_func(x)
                self.score.data = score_value
                self.mask = self.sparseness.get_mask(score_value)

    def forward(self, x):
        if not isinstance(self.sparseness, Dense):
            if self.training:
                self.set_score(x)
                self.mask = self.sparseness.get_mask(self.score)
                x, _ = Sparsifier.apply(
                    x, self.score, self.sparseness, self.backward_mode
                )
            else:
                x = x * self.mask
        if self.dump_to is not None:
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

    @property
    def weight_mask(self):
        return None if self.weight_sparsifier is None else self.weight_sparsifier.mask

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
