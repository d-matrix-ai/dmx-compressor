from torch.fx import Interpreter
from .utils import dmx_aware_functional_mappings
from torch.fx.node import Node
from typing import Any


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
            if (
                str(target) in dmx_aware_functional_mappings
                and dmx_aware_functional_mappings[str(target)].is_compound
            ):
                self.nodeInputs[n.name] = (args, kwargs)

            return getattr(self, n.op)(n.target, args, kwargs)
