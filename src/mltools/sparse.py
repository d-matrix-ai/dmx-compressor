from typing import Any
import torch
import torch.nn as nn
from parse import parse
from torch.autograd import Function
from mltools.utils.visualization import mask2braille

__ALL__ = [
    "Sparseness",
    "WeightSparseMixin",
    "Dense",
    "TopK",
    "BlockTopK",
    "Sparsify",
    "SparsificationManager",
]


class Sparseness(Function):
    r"""
    This class inherits from torch.autograd.Function, since when we use masking operations, we may have different forward and backward passes.
    Child classes to implement `get_mask()` and `from_shorthand()` method.
    """

    def __init__(self, mask_gradient=False):
        super().__init__()
        self.mask_gradient = torch.as_tensor(mask_gradient)

    def __str__(self) -> str:
        raise NotImplementedError

    def get_mask(self, *input: Any):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        # Propagrate None gradients to the parameters.
        # We'll need to run ctx.save_for_backward to save the two variables mask_gradient and mask during the forward pass.
        mask_gradient, mask = ctx.saved_tensors[:2]
        if mask_gradient.item():
            return grad_output * mask, None
        else:
            return grad_output, None

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

    def __init__(self, density=0.5, mask_gradient=False):
        super().__init__(mask_gradient)
        assert 0 <= density <= 1.0, "density has to be between 0 and 1"
        self.density = density

    @staticmethod
    def forward(ctx, score, params):
        (
            mask_gradient,
            density,
        ) = params
        _score = score.view(-1)
        idx = torch.argsort(_score, dim=0)[: int(score.numel() * (1.0 - density))]
        mask = (
            torch.ones_like(_score, device=score.device)
            .scatter_(dim=0, index=idx, value=0)
            .view_as(score)
        )
        ctx.save_for_backward(mask_gradient, mask)
        return mask

    def get_mask(self, score):
        params = (
            self.mask_gradient,
            self.density,
        )
        return TopK.apply(score, params)

    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse("TOPK{{{density:f}}}({mask_grad:l})", sh)
        return cls(
            density=conf["density"],
            mask_gradient=conf["mask_grad"] == "M",
        )

    def __str__(self) -> str:
        return f"Global TopK sparseness: density = {self.density}"

    def __repr__(self) -> str:
        return f"TOPK{{{self.density}}}({'M' if self.mask_gradient else 'U'})"


class BlockTopK(Sparseness):
    r"""
    Fine-grain structured sparsity with K non-zeros out of `block_size` elements along `block_dim`.
    """

    def __init__(self, K=4, block_size=8, block_dim=-1, mask_gradient=False):
        super().__init__(mask_gradient)
        assert 0 < K <= block_size, "N and M must be positive and N no greater than M"
        self.K = K
        self.block_size = block_size
        self.block_dim = block_dim

    @staticmethod
    def forward(ctx, score, params):
        mask_gradient, K, block_size, block_dim = params
        assert (
            score.shape[block_dim] % block_size == 0
        ), f"score has size {score.shape[block_dim]} at dimension {block_dim}, not a multiple of block size {block_size}"
        _score = score.transpose(block_dim, -1)
        score_shape = _score.shape
        _score = _score.reshape(-1, block_size)
        idx = torch.argsort(_score, dim=1)[:, : int(block_size - K)]
        mask = (
            torch.ones_like(_score, device=score.device)
            .scatter_(dim=1, index=idx, value=0)
            .reshape(score_shape)
            .transpose_(block_dim, -1)
        )
        ctx.save_for_backward(mask_gradient, mask)
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

    def get_mask(self, score):
        params = (self.mask_gradient, self.K, self.block_size, self.block_dim)
        return BlockTopK.apply(score, params)

    @classmethod
    def from_shorthand(cls, sh: str):
        conf = parse("BTOPK{{{K:d}:{block_size:d},{block_dim:d}}}({mask_grad:l})", sh)
        return cls(
            K=conf["K"],
            block_size=conf["block_size"],
            block_dim=conf["block_dim"],
            mask_gradient=conf["mask_grad"] == "M",
        )

    def __str__(self) -> str:
        return f"Block TopK sparseness: pattern = {self.K}:{self.block_size}, block dimension = {self.block_dim}"

    def __repr__(self) -> str:
        return f"BTOPK{{{self.K}:{self.block_size},{self.block_dim}}}({'M' if self.mask_gradient else 'U'})"


class Bernoulli(Sparseness):
    r"""Bernoulli sampler for supermasking."""

    @staticmethod
    def forward(ctx, score, params):
        # The scores need to be within [0, 1], otherwise there are bugs
        assert score.max() <= 1
        assert score.min() >= 0
        mask_gradient = params[0]
        mask = torch.bernoulli(score)
        ctx.save_for_backward(mask_gradient, mask)
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

    def get_mask(self, score):
        params = (self.mask_gradient,)
        return Bernoulli.apply(score, params)

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

    def __init__(
        self,
        tensor_shape,
        sparseness="DENSE",
        backward_mode="STE",
        score_func=None,
    ):
        super().__init__()

        self.score = nn.Parameter(torch.rand(tensor_shape), requires_grad=True)
        self.mask = torch.ones(tensor_shape)
        self.configure(sparseness, backward_mode, score_func)
        self.update_mask(self.score)
        self.plastic = False

    def configure(self, sparseness=None, backward_mode=None, score_func=None):
        # sparseness pattern
        if sparseness is not None:
            if not isinstance(sparseness, Sparseness):
                sparseness = Sparseness.from_shorthand(sparseness)
            if not hasattr(self, "sparseness") or repr(sparseness) != repr(
                self.sparseness
            ):
                self.sparseness = sparseness
        # backward mode
        if backward_mode is not None:
            self.backward_mode = backward_mode
            self.enable_weight_gradient = backward_mode.lower() in {"ste", "joint"}
            self.enable_mask_gradient = backward_mode.lower() in {"supermask", "joint"}
        # score function: importance <- score_func(score, input)
        if score_func is not None:
            self.score_func = score_func
            # mark the module as plastic for rewiring in the next forward() call
            self.plastic = True

    def update_mask(self, score):
        self.mask = self.sparseness.get_mask(score)

    def forward(self, x):
        if not isinstance(self.sparseness, Dense):
            if self.plastic:
                score = self.score_func(self.score, x)
                self.plastic = False
            else:
                score = self.score
            self.update_mask(score)
            if self.training:
                x = x if self.enable_weight_gradient else x.detach()
                self.mask = (
                    self.mask if self.enable_mask_gradient else self.mask.detach()
                )
            x = x * self.mask
        return x

    def mask_str(self, dims, max_elems):
        return mask2braille(self.mask, dims, max_elems)

   # def extra_repr(self):
    #    return f"sparseness = {self.sparseness.__repr__()}, backward_mode = {self.backward_mode}, mask = \n{self.mask_str(dims=(self.mask.ndim-2, self.mask.ndim-1), max_elems=32)}"


class SparsificationManager:
    r"""
    This is a sparsification management class
    Similar to Scheduler that manages parameter optimization through scheduler.step(),
    one can call sparsification_manager.step() to update underlying score.
    """

    def __init__(
        self,
        sparsify_modules,
        **kwargs,
    ):
        self.sparsify_modules = sparsify_modules

    def step(self, **kwargs):
        for sm in self.sparsify_modules:
            sm.configure(**kwargs)


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
