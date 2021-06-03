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
    "MagnitudeBasedPruning",
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
        if mode == "ste":
            return g_x, None
        elif mode == "supermask":
            return None, g_score
        elif mode == "joint":
            return g_x, g_score
        elif mode == "nm":
            return (
                g_x + 2e-4 * (1 - mask) * x,
                None,
            )  # https://github.com/NM-sparsity/NM-sparsity
        else:
            raise ValueError(f"unsupported backward mode: {mode}")

    @staticmethod
    def forward(ctx, x, score, sp, mode="ste"):
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
    Sparsification
    """

    def __init__(
        self, tensor_shape, sparseness=Dense(), backward_mode="ste", dump_to=None
    ):
        super().__init__()
        self.score = nn.Parameter(torch.ones(tensor_shape))
        self.sparseness = sparseness
        self.backward_mode = backward_mode
        self.dump_to = dump_to

    def forward(self, x):
        assert x.shape == self.score.shape, "score and x have to be of the same shape"
        if not isinstance(self.sparseness, Dense):
            x = Sparsifier.apply(x, self.score, self.sparseness, self.backward_mode)
        self.mask = self.sparseness.get_mask(self.score)
        if self.dump_to is not None:
            pass
        return x

    def extra_repr(self):
        return f"sparseness = {self.sparseness.__repr__()}, sparsifier = {self.sparsifier.__repr__()}"


class WeightSparseMixin:
    """
    Mixin for weight-sparse modules
    """

    def init_mask(self, sparsity=0.0):
        self.register_parameter("mask", nn.Parameter(torch.ones_like(self.weight)))
        if self.supermask:
            self.weight.requires_grad = False
        else:
            self.mask.requires_grad = False
        self.reset_mask(sparsity)

    def reset_mask(self, sparsity=0.0):
        """
        Reset the mask or supermask for sparse modules
        """
        if self.supermask:
            if self.deterministic_supermask:  # Ramanujan et al. 2019
                self.mask.data = torch.abs(self.weight.data)
            else:  # Radiya-Dixit & Wang 2020
                idx = torch.abs(self.weight).view(-1).argsort()
                init_val = 1.57  # 5
                init = torch.zeros_like(idx) + init_val
                init[idx[: math.floor(self.weight.numel() * sparsity)]] = -init_val
                self.mask.data = init.float().view_as(self.weight)
        else:  # static mask
            self.mask.data.bernoulli_(p=1.0 - sparsity)

    def configure_sparsity(
        self,
        sparse=False,
        supermask=False,
        deterministic_supermask=False,
        init_mask=False,
        sparsity=0.0,
    ):
        """
        Configure sparse states
        """
        self.sparse = sparse
        if supermask ^ self.supermask:  # change of supermask state
            self.supermask = supermask
            self.k = 1.0 - sparsity
            self.deterministic_supermask = deterministic_supermask
            self.weight.requires_grad = not supermask
            self.mask.requires_grad = supermask
            if init_mask:
                self.reset_mask(sparsity=sparsity)

    def get_mask(self):
        if self.supermask:
            mask = (
                EdgePopup.apply(self.mask, self.k)
                if self.deterministic_supermask
                else torch.sin(torch.clamp(self.mask, -math.pi / 2.0, math.pi / 2.0))
                # else Bernoulli.apply(torch.sigmoid(self.mask))
            )
        else:
            mask = self.mask
        return mask

    def get_mask_stats(self):
        """
        Returns the number of nonzero values in the mask and the total
        number of values in the mask.
        """
        assert self.sparse, "must be a weight sparse module"
        mask = self.get_mask()
        n_nonzero, n_total = int(mask.sum()), mask.numel()
        return n_nonzero, n_total

    @property
    def sparsity(self):
        if self.sparse:
            n_nonzero, n_total = self.get_mask_stats()
            return 1.0 - float(n_nonzero) / n_total
        else:
            return None

    def prune_by_threshold(self, threshold=0.0, zero_pruned=False):
        """
        Prune weight by a threshold
        Args:
            threshold (float): pruning threshold (default: 0)
            zero_pruned (bool): whether to set pruned weights to 0 in addition to setting mask to 0 (default: False)
        """
        if self.sparse and not self.supermask:
            to_be_pruned = torch.abs(self.weight * self.mask) < threshold
            self.mask *= 1 - to_be_pruned.float()
            if zero_pruned:
                self.weight.data *= self.mask

    def prune_to_sparsity(self, sparsity=0.0, zero_pruned=False):
        """
        Prune weight to at least a sparsity level
        Args:
            sparsity (float): target sparsity (default: 0)
            zero_pruned (bool): whether to set pruned weights to 0 in addition
                                to setting mask to 0 (default: False)
        """
        if self.sparse and not self.supermask:
            idx = torch.abs(self.weight * self.mask).view(-1).argsort()
            to_be_pruned = torch.zeros_like(idx)
            to_be_pruned[idx[: math.floor(self.weight.numel() * sparsity)]] = 1
            self.mask *= 1 - to_be_pruned.float().view_as(self.mask)
            if zero_pruned:
                self.weight.data *= self.mask

    @property
    def effective_weight(self):
        "Effective weight, i.e. masked weight in the sparse case"
        return self.weight * self.get_mask() if self.sparse else self.weight


class Linear(nn.Linear, WeightSparseMixin):
    """
    Wrapper of torch.nn.Linear
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        sparse=False,
        supermask=False,
        deterministic_supermask=False,
        sparsity=0.0,
    ):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.sparse = sparse
        self.supermask = supermask
        self.deterministic_supermask = deterministic_supermask
        self.k = 1.0 - sparsity
        self.init_mask(sparsity=0.0)

    def forward(self, input):
        return F.linear(input, self.effective_weight, self.bias)

    def extra_repr(self):
        return (
            "in_features={}, out_features={}, bias={}, sparse={}, supermask={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.sparse,
                self.supermask,
            )
        )


class Embedding(nn.Embedding, WeightSparseMixin):
    """
    Wrapper of torch.nn.Embedding
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        _weight=None,
        sparse=False,
        supermask=False,
        deterministic_supermask=False,
        sparsity=0.0,
    ):
        super(Embedding, self).__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
        )
        self.sparse = sparse
        self.supermask = supermask
        self.deterministic_supermask = deterministic_supermask
        self.k = 1.0 - sparsity
        self.init_mask(sparsity=sparsity)

    def forward(self, input):
        return F.embedding(
            input,
            self.effective_weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            False,
        )

    def extra_repr(self):
        return "num_embeddings={}, embedding_dim={}, padding_idx={}, max_norm={}, norm_type={}, scale_grad_by_freq={}, sparse={}, supermask={}".format(
            self.num_embeddings,
            self.embedding_dim,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
            self.supermask,
        )


class Conv2d(nn.Conv2d, WeightSparseMixin):
    r"""
    Wrapper of torch.nn.Conv2d
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        sparse=False,
        supermask=False,
        deterministic_supermask=False,
        parsity=0.0,
    ):
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.sparse = sparse
        self.supermask = supermask
        self.deterministic_supermask = deterministic_supermask
        self.k = 1.0 - sparsity
        self.init_mask(sparsity=sparsity)

    def forward(self, input):
        return self.conv2d_forward(input, self.effective_weight)

    def extra_repr(self):
        return "in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, dilation={}, groups={}, bias={}, padding_mode={}, sparse={}, supermask={}".format(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias,
            self.padding_mode,
            self.sparse,
            self.supermask,
        )


if __name__ == "__main__":
    pass
