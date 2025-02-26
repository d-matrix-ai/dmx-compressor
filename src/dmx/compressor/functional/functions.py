import torch
from types import SimpleNamespace


__ALL__ = ["experimental"]

class experimental(SimpleNamespace):
    def gelu(
        input: torch.Tensor, approximate: str = "none", **extra_params
    ) -> torch.Tensor:
        r"""
        An absurd approximation of the gelu function
        """
        # 1. parse extra_params
        _scale = extra_params["scale"]
        # 2. resolve paramter conflicts
        assert (
            approximate == "none"
        ), "approximate has to be 'none', not functionally meaningful anyway"
        # 3. custom logic
        return torch.nn.functional.relu(input.to(torch.float16)) * _scale
