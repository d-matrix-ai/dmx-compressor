import sys
import torch
from .nn import *
from mltools import dmir
from mltools.utils import load_config_file


class Model(torch.nn.Module):
    r"""
    Container for a DNN model to be deployed
    - body to be mapped on device
    - head and tail to be executed on host, corresponding to pre- and post-processing
    - equipped with corsair-aware transformation
    """

    def __init__(
        self, body, head=torch.nn.Identity(), tail=torch.nn.Identity()
    ) -> None:
        super().__init__()
        self.body = body
        self.head = head
        self.tail = tail

    def forward(self, input):
        # NOTE: only a single input is allowed
        return self.tail(self.body(self.head(input)))

    def transform(self, config="configs/corsair.yaml"):
        r"""
        Transform with Corsair-sprcific numerics/sparsity/logics
        """
        if isinstance(config, str):
            config = load_config_file(config)

        for n, m in self.body.named_modules():
            if isinstance(m, CorsairModule):
                for r in config["transformation_rules"]:
                    if (
                        isinstance(m, eval(r["instance"]))
                        and all([_n in n for _n in r["name_includes"]])
                        and all([not _n in n for _n in r["name_excludes"]])
                    ):
                        m._transform(r["config"])

    def dmir_graph(self, input, **kwargs):
        return dmir.dump(
            self.body,
            self.head(input),
            **kwargs,
        )
