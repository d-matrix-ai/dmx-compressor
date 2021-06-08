from typing import Union, List, Tuple
import sys
import torch
from torch import Tensor, Size
import torch.nn.functional as F
from numerical import (
    Format,
    BoundaryCastMixin,
    Same,
    FixedPoint,
    FloatingPoint,
    BlockFloatingPoint,
    CastTo,
)
from sparse import (
    Sparseness,
    WeightSparseMixin,
    Dense,
    TopK,
    BlockTopK,
    Bernoulli,
    Sparsify,
)
from utils import load_config_file

__ALL__ = ["nn", "transform", "CorsairConfig"]


class CorsairModule(torch.nn.Module):
    r"""
    Model container equipped with corsair transform
    """

    def __init__(self):
        super().__init__()

    def transform(self, config_file="configs/corsair.yaml"):
        r"""
        Model conversion for Corsair numerics/sparsity simulation/optimization
        """
        config = load_config_file(config_file)

        for n, m in self.named_modules():
            for r in config["transformation_rules"]:
                if (
                    isinstance(m, getattr(sys.modules[__name__], r["instance"]))
                    and all([_n in n for _n in r["name_includes"]])
                    and all([not _n in n for _n in r["name_excludes"]])
                ):
                    m.transform(r["config"])
        print(self)


class CorsairMixin(BoundaryCastMixin, WeightSparseMixin):
    r"""
    Extending torch.nn.Module
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def transform(self, config):
        # numerics transformation
        self.input_cast.format = Format.str2format(config["input_format"])
        self.output_cast.format = Format.str2format(config["output_format"])
        if self.accum_cast is not None:
            self.accum_cast.format = Format.str2format(config["accum_format"])
        if self.weight_cast is not None:
            self.weight_cast.format = Format.str2format(config["weight_format"])
        if self.bias_cast is not None:
            self.bias_cast.format = Format.str2format(config["bias_format"])
        # sparsity transformation
        if self.weight_sparsifier is not None:
            self.weight_sparsifier.sparseness = Sparseness.str2sparseness(config["weight_sparseness"])
            # TODO: need to figure out a better way of handling score setting
            self.weight_sparsifier.set_score(torch.abs(self.weight))


class Linear(CorsairMixin, torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, input: Tensor) -> Tensor:
        _input = self.input_cast(input)
        _weight = self.weight_cast(self.effective_weight)
        _product = self.accum_cast(F.linear(_input, _weight, None))
        _bias = self.bias_cast(self.bias)
        _output = torch.add(_product, _bias)
        output = self.output_cast(_output)
        return output


class Softmax(CorsairMixin, torch.nn.Softmax):
    def __init__(self, dim: int = -1) -> None:
        super().__init__(dim=dim)

    def forward(self, input: Tensor) -> Tensor:
        _output = self.input_cast(input)
        _output = super().forward(_output)
        output = self.output_cast(_output)
        return output


class LayerNorm(CorsairMixin, torch.nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine
        )

    def forward(self, input: Tensor) -> Tensor:
        _input = self.input_cast(input)
        _weight = self.weight_cast(self.weight)
        _bias = self.bias_cast(self.bias)
        output = F.layer_norm(_input, self.normalized_shape, _weight, _bias, self.eps)
        return output


class Dropout(CorsairMixin, torch.nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p=p, inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        _output = self.input_cast(input)
        _output = super().forward(_output)
        output = self.output_cast(_output)
        return output


# overload torch.nn modules
nn = torch.nn
nn.Module = CorsairModule
nn.Linear = Linear
nn.Softmax = Softmax
nn.LayerNorm = LayerNorm
nn.Dropout = Dropout


if __name__ == "__main__":
    pass
