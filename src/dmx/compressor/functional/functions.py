import torch
from types import SimpleNamespace


__ALL__ = ["experimental"]

class experimental(SimpleNamespace):
    def silu(
        input: torch.Tensor, inplace: bool = False, **extra_params
    ) -> torch.Tensor:
        r"""
        An absurd approximation of the silu function
        """
        # 1. parse extra_params
        _scale = extra_params["scale"]
        # 2. resolve paramter conflicts
        assert (
            not inplace
        ), "inplace has to be False, not functionally meaningful anyway"
        # 3. custom logic
        return torch.nn.functional.relu(input.to(torch.float16)) * _scale
