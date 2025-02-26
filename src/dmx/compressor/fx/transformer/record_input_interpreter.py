from torch.fx import Interpreter
from .utils import dmx_aware_functional_mappings
from torch.fx.node import Node
from typing import Any
import torch


class RecordInputInterpreter(Interpreter):
    """
    Interpreter that captures the input tensors to the compound functions.
    attributes:
        nodeInputs: dictionary that maps node name to its tensor args and kwargs
    """

    def __init__(self, module, garbage_collect_values=True, graph=None):
        super().__init__(module, garbage_collect_values, graph)
        self.nodeInputs = {}

    def run_node(self, n: Node) -> Any:
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Record input tensors to nodeInputs if target is in compound_functions

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        with self._set_current_node(n):
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)

            target = n.target

            # recording inputs
            if str(target) in dmx_aware_functional_mappings and (
                dmx_aware_functional_mappings[str(target)].is_compound
                or str(target)
                in (
                    "<built-in function mul>",
                    "<built-in function add>",
                )
            ):
                self.nodeInputs[n.name] = (args, kwargs)

            device = None
            if (
                target in self.submodules
                and hasattr(self.submodules[target], "weight")
                and self.submodules[target].weight is not None
            ):
                device = self.submodules[target].weight.device
            for a in args:
                if isinstance(a, torch.Tensor) and device is None:
                    device = a.device
            if device is not None:
                args = tuple(
                    a.to(device) if isinstance(a, torch.Tensor) else a for a in args
                )
                kwargs = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                }
            return getattr(self, n.op)(n.target, args, kwargs)
