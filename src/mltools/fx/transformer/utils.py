from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, Set
import torch.fx as fx
from torch.fx.proxy import Proxy
import re
import torch

from mltools import dmx

dmx_aware_mapping = {
    "torch.nn.modules.sparse.Embedding": dmx.nn.Embedding,
    "torch.nn.modules.linear.Linear": dmx.nn.Linear,
    "torch.nn.modules.conv.Conv1d": dmx.nn.Conv1d,
    "torch.nn.modules.conv.Conv2d": dmx.nn.Conv2d,
    "torch.nn.modules.pooling.AdaptiveAvgPool2d": dmx.nn.AdaptiveAvgPool2d,
    "torch.nn.modules.pooling.MaxPool2d": dmx.nn.MaxPool2d,
    "torch.nn.modules.batchnorm.BatchNorm2d": dmx.nn.BatchNorm2d,
    "torch.nn.modules.normalization.LayerNorm": dmx.nn.LayerNorm,
    "torch.nn.modules.dropout.Dropout": dmx.nn.Dropout,
    "torch.nn.modules.activation.Softmax": dmx.nn.Softmax,
    "torch.nn.modules.activation.ReLU": dmx.nn.ReLU,
    "torch.nn.modules.activation.ReLU6": dmx.nn.ReLU6,
    "torch.nn.modules.activation.SiLU": dmx.nn.SiLU,
    "torch.nn.modules.activation.Tanh": dmx.nn.Tanh,
    "torch.nn.modules.activation.GELU": dmx.nn.GELU,
    "transformers.pytorch_utils.Conv1D": dmx.nn.Linear,
    "transformers.activations.NewGELUActivation": dmx.nn.GELU,
    "transformers.activations.GELUActivation": dmx.nn.GELU,
    "transformers.activations.FastGELUActivation": dmx.nn.GELU,
    "transformers.activations.QuickGELUActivation": dmx.nn.GELU,
    "transformers.activations.ClippedGELUActivation": dmx.nn.GELU,
    "transformers.models.bloom.modeling_bloom.BloomGelu": dmx.nn.GELU,
    "transformers.activations.SiLUActivation": dmx.nn.SiLU,
    "transformers.models.t5.modeling_t5.T5LayerNorm": dmx.nn.RMSNorm,
    "transformers.models.llama.modeling_llama.LlamaRMSNorm": dmx.nn.RMSNorm,
}

dmx_aware_functional_mappings = {
    "torch.nn.functional.relu": dmx.nn.ReLU,
    "torch.nn.functional.relu6": dmx.nn.ReLU6,
    "torch.nn.functional.silu": dmx.nn.SiLU,
    "torch.nn.functional.tanh": dmx.nn.Tanh,
    "torch.nn.functional.gelu": dmx.nn.GELU,
    "torch.nn.functional.softmax": dmx.nn.Softmax,
    "torch.nn.functional.dropout": dmx.nn.Dropout,
    "torch.matmul": dmx.nn.ActActMatMul,
}
for f_key in list(dmx_aware_functional_mappings.keys()):
    new_key = repr(eval(f_key))
    dmx_aware_functional_mappings[new_key] = dmx_aware_functional_mappings.pop(f_key)
dmx_aware_functional_mappings["<built-in function add>"] = dmx.nn.ResAdd


def process_args(args):
    """
    A function that goes through each arg in args and removes the proxy wrapper.

    Args:
        args (Tuple): args for which proxy wrappers should be removed

    Returns:
        A tuple of args where each element has the proxy wrapper removed

    """
    new_args = ()
    for arg in args:
        if isinstance(arg, Proxy):
            new_args += (arg.node,)
        else:
            new_args += (arg,)
    return new_args


def process_kwargs(kwargs):
    """
    A function that goes through each kwarg in kwargs and removes the proxy wrapper.

    Args:
        kwargs (Dict): kwargs for which proxy wrappers should be removed

    Returns:
        A dictionary of kwargs where each element has the proxy wrapper removed

    """
    new_kwargs = dict()
    for k in kwargs.keys():
        if isinstance(kwargs[k], Proxy):
            new_kwargs[k] = kwargs[k].node
        else:
            new_kwargs[k] = kwargs[k]
    return new_kwargs


def get_name_for_func_nodes(
    candidate: str, used_names: Set[str], base_count: Dict[str, int]
):
    """Get the unique name for functional nodes

    Arguments:
        candidate (str): used as the basis for the unique name, relevant to the user.
        used_names (Set[str]): A Set of names already used for nodes
        base_count (Dict[str, int]): A dict counting number of names sharing the same candidate base
    """
    illegal_char_regex = re.compile("[^0-9a-zA-Z_]+")
    name_suffix_regex = re.compile(r"(.*)_(\d+)$")

    candidate = illegal_char_regex.sub("_", candidate)
    if not candidate:
        candidate = "_unnamed"

    if candidate[0].isdigit():
        candidate = f"_{candidate}"

    match = name_suffix_regex.match(candidate)
    if match is None:
        base = candidate
        num = None
    else:
        base, num_str = match.group(1, 2)
        num = int(num_str)

    candidate = base if num is None else f"{base}_{num}"
    if not num:
        num = base_count[base]

    while candidate in used_names or fx.graph._Namespace()._is_illegal_name(
        candidate, None
    ):
        num += 1
        candidate = f"{base}_{num}"
    return candidate


def _extract_ops(gm: torch.fx.GraphModule):
    r"""
    A generator that gathers ops from a GraphModule
    """
    for _op in gm.graph.nodes:
        if _op.op == "call_module":
            yield _op.name, (gm.get_submodule(_op.target).__class__)
        if _op.op == "call_method":
            yield _op.name, getattr(torch.Tensor, _op.target)
        if _op.op == "call_function":
            yield _op.name, _op.target


def get_op_set_from(gm: torch.fx.GraphModule):
    r"""
    Returns a set of ops from a GraphModule
    """
    return set(t for _, t in _extract_ops(gm))
