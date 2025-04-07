from abc import abstractmethod
from typing import Optional
from types import SimpleNamespace
import torch
from torch import Tensor
from torch.fx import Graph

from dmx.compressor.numerical import NumericalCastMixin, Same, CastTo
from dmx.compressor.sparse import (
    WeightSparseMixin,
    Dense,
    LazySparsify,
)
from dmx.compressor.functional import (
    ApproximationMixin,
    NoApproximation,
)
from dmx.compressor.perf_proxy import PerformanceProxyMixin
from dmx.compressor.layer_reconstruction import LayerReconstructionMixin


__ALL__ = ["DmxModuleType", "DmxModule", "DmxModuleConfig", "is_configurable"]


class DmxModuleType(type):
    pass


class DmxModule(
    ApproximationMixin,
    PerformanceProxyMixin,
    LayerReconstructionMixin,
    NumericalCastMixin,
    WeightSparseMixin,
    torch.nn.Module,
    metaclass=DmxModuleType,
):
    r"""
    Extended torch.nn.Module for Dmx to support quantization.

    Args:
        *args (Optional[Tuple]): variable length of args
        state_dict_url (Optional[str]): Url for loading the state dicts. Defaults to None.
        **kwargs (Optional[Dict]): variable length of kwargs

    Attributes:
        state_dict_url (str): Url for loading the module state dicts.
    """

    is_compound = False
    functional_forward = None

    def __init__(self, *args, state_dict_url: Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state_dict_url = state_dict_url
        if state_dict_url is not None:
            self.load_state_dict_and_register_url(state_dict_url)

    def configure(self, config) -> None:
        """
        A function that changes the format of the ops and loading state dicts according to the config file.

        Args:
            config (DmxModuleConfig): config file for setting new formats and loading state dicts.

        """
        # numerics transformation
        if "input_formats" in config:
            self.input_casts.set_format(format=config["input_formats"])
        if "output_formats" in config:
            self.output_casts.set_format(format=config["output_formats"])
        if self.accum_cast is not None and "accum_format" in config:
            self.accum_cast.set_format(format=config["accum_format"])
        if self.weight_storage_cast is not None and "weight_storage_format" in config:
            self.weight_storage_cast.set_format(format=config["weight_storage_format"])
        if self.weight_cast is not None and "weight_format" in config:
            self.weight_cast.set_format(format=config["weight_format"])
        if self.bias_cast is not None and "bias_format" in config:
            self.bias_cast.set_format(format=config["bias_format"])
        if self.smoothquant is not None and "smoothquant_scale_format" in config:
            self.smoothquant.set_scale_format(format=config["smoothquant_scale_format"])
        # sparsity transformation
        if self.weight_sparsifier is not None and "weight_sparseness" in config:
            self.weight_sparsifier.configure(sparseness=config["weight_sparseness"])
        # custom logic transformation
        if "approximation_function" in config:
            self.approximator.set_function(config["approximation_function"])

        # load state_dict if provided
        if (
            "state_dict_url" in config
            and config["state_dict_url"] != self.state_dict_url
        ):
            self.load_state_dict_and_register_url(config["state_dict_url"])

    transform = configure  # NOTE: to be deprecated

    def load_state_dict_and_register_url(self, url: str) -> None:
        """
        A function that loads state dict from a url and sets url to self.state_dict_url

        Args:
            ulr (str): url for loading the state dict
        """
        self.load_state_dict(torch.hub.load_state_dict_from_url(url))
        self.state_dict_url = url

    def save_state_dict_and_register_url(self, parent_dir: str) -> None:
        """
        A function that saves the current state dict of the module to a url under a specified parent directory

        Args:
            parent_dir (str): parent directory for the url
        """
        from dmx.compressor.utils import save_state_dict_and_return_url

        url = save_state_dict_and_return_url(self, parent_dir)
        self.state_dict_url = url

    def dmx_config(self, freeze=False):
        """
        A function that the DmxModuleConfig object for the module

        Args:
            freeze (bool): if True, both state dict and ops formats would be included in the returned DmxModuleConfig. If False, only state dict will be included.

        Returns:
            A DmxModuleConfig object for the module
        """
        return DmxModuleConfig.from_module(self, freeze)

    def fold_weight_and_bias(self) -> None:
        """
        A function that applies the ops the weights and biases using the corresponding formats.
        """
        with torch.no_grad():
            # bias cast
            if self.bias_cast is not None and not isinstance(self.bias_format, Same):
                self.bias.data = self.bias_cast(self.bias.data)
                self.bias_cast = CastTo(format=Same())
            # weight sparsifier
            if self.weight_sparsifier is not None and not isinstance(
                self.weight_sparseness, Dense
            ):
                self.weight.data = self.effective_weight
                self.weight_sparsifier = LazySparsify(sparseness=Dense())
            # weight smoothquant
            if (
                self.smoothquant is not None
                and self.smoothquant.fused_to_weight[0] == 0
            ):
                self.smoothquant.fuse_to_weight(self.weight)
            # weight storage cast
            if self.weight_storage_cast is not None and not isinstance(self.weight_storage_cast, Same):
                self.weight.data = self.weight_storage_cast(self.weight.data)
                self.weight_storage_cast = CastTo(format=Same())
            # weight cast
            if self.weight_cast is not None and not isinstance(self.weight_cast, Same):
                self.weight.data = self.weight_cast(self.weight.data)
                self.weight_cast = CastTo(format=Same())

    @property
    def weight_hypernet(self):
        """
        Returns a function that processes weight according to the ops format of the module
        """

        def _weight_hypernet(_w):
            if self.weight_sparsifier is not None:
                _w = self.weight_sparsifier(_w)
            if (
                self.smoothquant is not None
                and self.smoothquant.fused_to_weight[0] == 0
            ):
                _w = self.smoothquant.scale_weight(_w)
            if self.weight_storage_cast is not None:
                _w = self.weight_storage_cast(_w)
            if self.weight_cast is not None:
                _w = self.weight_cast(_w)
            return _w

        return _weight_hypernet

    @property
    def _weight(self):
        """
        Returns the quantized weights of the module
        """
        return self.weight_hypernet(self.weight)

    @property
    def _bias(self):
        """
        Returns the quantized bias of the module
        """

        return self.bias_cast(self.bias) if self.bias_cast is not None else None

    def forward(self, input: Tensor, *args, **kwags) -> Tensor:
        """
        Forward pass of the module with quantization ops applied.

        Args:
            input (Tensor): input tensor to be passed through the module
            *args (Optional[Tuple]): variable length of args
            **kwargs (Optional[Dict]): variable length of kwargs
        """
        _dtype, _device = input.dtype, input.device
        if hasattr(self, "weight") and self.weight is not None:
            _device = self.weight.device
        if self.smoothquant is not None:
            if self.smoothquant.dynamic[0] == 1 or self.smoothquant.calibrating:
                self.update_smoothquant_scale(input)
            input = self.smoothquant.scale_input(input)
        _input, args, kwags = self.input_casts(input, *args, **kwags)
        if self.obc is not None:
            self.obc.measure_hessian(_input)
        _input, args, kwags = self.align_device(_input, args, kwags, _device)
        _output = self._forward(_input, *args, **kwags)
        output = self.output_casts(_output, output=True)
        if self.flop_counter_enabled:
            self.count_flops(input, output)
        if self.align_boundary_dtype:
            output = (
                type(output)(a.to(_dtype) for a in output)
                if isinstance(output, (tuple, list))
                else output.to(_dtype)
            )
        return output

    def align_device(self, _input, args, kwags, _device):
        _input = _input.to(_device) if _input.device != _device else _input
        args = tuple(
            x.to(_device) if isinstance(x, torch.Tensor) and x.device != _device else x
            for x in args
        )
        for k in kwags.keys():
            if isinstance(kwags[k], Tensor) and kwags[k].device != _device:
                kwags[k] = kwags[k].to(_device)
        return _input, args, kwags

    def update_params_with_raw(self, raw: torch.nn.Module) -> None:
        """
        Update parameters of a DmxModule from a torch.nn.Module.

        Args:
            raw (torch.nn.Module): the torch module to copy parameters from.
        """
        self.load_state_dict(raw.state_dict(), assign=True)
        # Inherit device from raw module
        for n, m in raw.named_parameters():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            device = m.device if m.device != "cpu" else device
            setattr(self, n, torch.nn.Parameter(getattr(self, n).to(device)))
        # inherit some module attributes from raw module
        self.training = raw.training
        if hasattr(raw, "dtype"):
            self.dtype = raw.dtype

    @abstractmethod
    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        raise NotImplementedError("to_compiler_graph not implemented!")

    def create_placeholders(self, g, names):
        placeholder_nodes = []
        for name in names:
            _n = g.placeholder(name)
            placeholder_nodes.append(_n)
        return placeholder_nodes

    def qdq_nodes(self, g, nodes, cast_names):
        from operator import attrgetter

        dq_nodes = []
        for node, cast_name in zip(nodes, cast_names):
            _scale = g.get_attr(f"{cast_name}.scale")
            _zero_point = g.get_attr(f"{cast_name}.zero_point")
            _q = g.call_function(
                torch.ops.dmx_ops.quantize,
                (
                    node,
                    _scale,
                    _zero_point,
                    repr(attrgetter(f"{cast_name}.format")(self)),
                ),
            )
            _dq = g.call_function(
                torch.ops.dmx_ops.dequantize,
                (_q, _scale, _zero_point),
            )
            dq_nodes.append(_dq)
        if len(dq_nodes) == 1:
            return dq_nodes[0]
        return dq_nodes


class DmxModuleConfig(dict):
    @classmethod
    def from_module(cls, module: DmxModule, freeze=False):
        """
        A function that stores state and ops format of the module in a DmxModuleConfig object

        Args:
            module (DmxModule): Target module for creating the DmxModuleConfig

        Returns:
            A DmxModuleConfig object that stores state and ops format of the module in a DmxModuleConfig object
        """
        cc = SimpleNamespace()
        cc.instance_of = module.__class__
        if isinstance(module, DmxModule):
            if module.input_formats is not None and (
                freeze
                or not all(isinstance(f, Same) for f in module.input_formats.values())
            ):
                cc.input_formats = module.input_formats
            if module.output_formats is not None and (
                freeze
                or not all(isinstance(f, Same) for f in module.input_formats.values())
            ):
                cc.output_formats = module.output_formats
            if module.accum_format is not None and (
                freeze or not isinstance(module.accum_format, Same)
            ):
                cc.accum_format = module.accum_format
            if module.weight_format is not None and (
                freeze or not isinstance(module.weight_format, Same)
            ):
                cc.weight_format = module.weight_format
            if module.weight_storage_format is not None and (
                freeze or not isinstance(module.weight_storage_format, Same)
            ):
                cc.weight_storage_format = module.weight_storage_format
            if module.bias_format is not None and (
                freeze or not isinstance(module.bias_format, Same)
            ):
                cc.bias_format = module.bias_format
            if module.smoothquant is not None and (
                freeze or not isinstance(module.smoothquant.scale_cast.format, Same)
            ):
                cc.smoothquant_scale_format = module.smoothquant.scale_cast.format
            if module.weight_sparseness is not None and (
                freeze or not isinstance(module.weight_sparseness, Dense)
            ):
                cc.weight_sparseness = module.weight_sparseness
            if freeze or not isinstance(module.approximation_function, NoApproximation):
                cc.approximation_function = module.approximation_function
            if module.state_dict_url is not None:
                cc.state_dict_url = module.state_dict_url
        return cc.__dict__


def is_configurable(m):
    return isinstance(m, DmxModule)
