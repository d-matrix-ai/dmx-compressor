import torch
from types import SimpleNamespace


__ALL__ = ["crazy"]

class crazy(SimpleNamespace):
    def gelu(
        input: torch.Tensor, approximate: str = "none", **extra_params
    ) -> torch.Tensor:
        r"""
        An absurd approximation of the gelu function
        """
        assert (
            approximate == "none"
        ), "approximate has to be 'none', not functionally meaningful anyway"
        _scale = extra_params["scale"]
        return torch.nn.functional.relu(input) * _scale
