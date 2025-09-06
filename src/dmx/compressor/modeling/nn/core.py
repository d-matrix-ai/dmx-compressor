from abc import abstractmethod
import time
from typing import Optional,List
from contextlib import contextmanager
from types import SimpleNamespace
import torch
from torch import Tensor
from torch.fx import Graph
import inspect

from dmx.compressor.numerical import NumericalCastMixin, Same, CastTo
from dmx.compressor.plugins import PluginLayerData,PluginBase
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

import warnings

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
    plugins : List[PluginBase] = []
    
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
        if "pre_input_transform" in config:
            self.input_casts.set_pre_transform(config["pre_input_transform"])
        if "output_formats" in config:
            self.output_casts.set_format(format=config["output_formats"])
        if "pre_output_transform" in config:
            self.output_casts.set_pre_transform(config["pre_output_transform"])
            
        if self.accum_cast is not None and "accum_format" in config:
            self.accum_cast.set_format(format=config["accum_format"])
        if self.weight_storage_cast is not None and "weight_storage_format" in config:
            self.weight_storage_cast.set_format(format=config["weight_storage_format"])
        if self.weight_cast is not None and "weight_format" in config:
            self.weight_cast.set_format(format=config["weight_format"])
        if self.weight_cast is not None and "pre_weight_transform" in config:
            self.weight_cast.set_pre_transform(config["pre_weight_transform"])
            
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
            if self.weight_storage_cast is not None and not isinstance(
                self.weight_storage_cast, Same
            ):
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

    def forward(self, input: Tensor, *args, **kwargs) -> Tensor:
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
        _input, args, kwargs = self.input_casts(input, *args, **kwargs)
        if self.obc is not None:
            self.obc.measure_hessian(_input)
        _input, args, kwargs = self.align_device(_input, args, kwargs, _device)
        if self.aft is not None:
            self.aft.optimize(_input, *args, **kwargs)
        _output = self._forward(_input, *args, **kwargs)
        output = self.output_casts(_output, output=True)
        plugin_data = PluginLayerData(input_before_cast = input,
                                      input_after_cast = _input,
                                      output_before_cast = _output,
                                      output_after_cast = output,
                                      mod = self,
                                      args = args,
                                      kwargs = kwargs)
        plugins_copy = DmxModule.plugins.copy()
        for p in plugins_copy:
            #To avoid infinite recursion if the plugin calls forward on the DmxModule
            DmxModule.plugins.remove(p)
            p.process_layer(plugin_data)
            DmxModule.plugins = plugins_copy.copy()
            
        if self.flop_counter_enabled:
            self.count_flops(input, output)
        if self.align_boundary_dtype:
            output = (
                type(output)(a.to(_dtype) for a in output)
                if isinstance(output, (tuple, list))
                else output.to(_dtype)
            )
        return output

    def align_device(self, _input, args, kwargs, _device):
        _input = _input.to(_device) if _input.device != _device else _input
        args = tuple(
            x.to(_device) if isinstance(x, torch.Tensor) and x.device != _device else x
            for x in args
        )
        for k in kwargs.keys():
            if isinstance(kwargs[k], Tensor) and kwargs[k].device != _device:
                kwargs[k] = kwargs[k].to(_device)
        return _input, args, kwargs

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

    @contextmanager
    def monitoring(self, _records: list):
        """
        Context manager for monitoring input/output to/from the DmxModule
        """

        def recorder(_mod, _inp, _input_kwargs, _out) -> None:
            _records.append(
                dict(
                    input=(_inp, _input_kwargs),
                    output=_out,
                )
            )

        _h = self.register_forward_hook(recorder, with_kwargs=True)
        yield self
        _h.remove()

    @contextmanager
    def measuring_runtime(self, _records: list, device: torch.device):
        """
        Context manager for monitoring runtime of DmxModule
        """

        def tick(_mod, _inp):
            assert len(_records) == 0 or not isinstance(
                _records[-1], list
            ), f"multiple tick calls without tock in between: {_records}"
            if device.type == "cuda":
                start_t = torch.cuda.Event(enable_timing=True)
                start_t.record()
            else:
                start_t = time.perf_counter()
            _records.append([start_t])

        def tock(_mod, _inp, _out):
            assert len(_records) > 0 and isinstance(
                _records[-1], list
            ), f"tock called without preceding tick : {_records}"
            # start_t can be float(time) or cuda event
            start_t = _records.pop()[0]
            if device.type == "cuda":
                end_t = torch.cuda.Event(enable_timing=True)
                end_t.record()
                # The synchronization slows things down. We might need to wait till
                # the model finishes running to synchronize and gather the event deltas
                torch.cuda.synchronize()
                _records.append(start_t.elapsed_time(end_t) / 1000)

            else:
                end_t = time.perf_counter()
                _records.append(end_t - start_t)

        _h_pre = self.register_forward_pre_hook(tick)
        _h_post = self.register_forward_hook(tock)
        yield self
        _h_pre.remove()
        _h_post.remove()

    @abstractmethod
    def to_compiler_graph(self) -> Graph:
        """
        Returns a compiler friendly graph
        """
        raise NotImplementedError("to_compiler_graph not implemented!")


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


class DmxGraph(Graph):
    def __init__(self, owning_module=None, tracer_cls=None, tracer_extras=None):
        super().__init__(owning_module, tracer_cls, tracer_extras)
        try:
            import dmx.ops
        except:
            warnings.warn("Falling back to dummy q/dq torch ops")

            @torch.library.custom_op("dmx_ops::quantize.Scalar", mutates_args=[])
            def quantize(
                t: float, scale: float, zero_point: float, format: str
            ) -> torch.Tensor:
                return torch.Tensor([t])

            @torch.library.custom_op("dmx_ops::quantize.Tensor", mutates_args=[])
            def quantize(
                t: torch.Tensor, scale: float, zero_point: float, format: str
            ) -> torch.Tensor:
                return t.clone()

            @quantize.register_fake
            def _(
                t: torch.Tensor, scale: float, zero_point: float, format: str
            ) -> torch.Tensor:
                return torch.empty_like(t)

            @torch.library.custom_op("dmx_ops::dequantize.Tensor", mutates_args=[])
            def dequantize(
                t: torch.Tensor, scale: float, zero_point: float
            ) -> torch.Tensor:
                return t.clone()

            @dequantize.register_fake
            def _(t: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
                return torch.empty_like(t)

    def qdq_node(self, node, cast_name, cast_format):
        from operator import attrgetter

        if node is None or cast_name is None:
            return node

        _scale = self.get_attr(f"{cast_name}.scale")
        _zero_point = self.get_attr(f"{cast_name}.zero_point")
        _q = self.call_function(
            torch.ops.dmx_ops.quantize,
            (
                node,
                _scale,
                _zero_point,
                cast_format,
            ),
        )
        dq_node = self.call_function(
            torch.ops.dmx_ops.dequantize,
            (_q, _scale, _zero_point),
        )
        return dq_node

    def create_placeholders(self, names, cast_names=None, cast_formats=None):
        placeholder_nodes = []
        if cast_names is None:
            cast_names = [None] * len(names)
            cast_formats = [None] * len(names)
        assert len(cast_names) == len(
            names
        ), "List of placeholder names and cast names shuold have same length!"
        for name, c_name, c_format in zip(names, cast_names, cast_formats):
            _n = self.placeholder(name, c_name, c_format)
            placeholder_nodes.append(_n)
        if len(placeholder_nodes) == 1:
            return placeholder_nodes[0]
        return placeholder_nodes

    def placeholder(
        self,
        name,
        cast_name=None,
        cast_format=None,
        type_expr=None,
        default_value=inspect.Signature.empty,
    ):
        node = super().placeholder(name, type_expr, default_value)
        node = self.qdq_node(node, cast_name, cast_format)
        return node

    def get_attr(
        self,
        qualified_name,
        cast_name=None,
        cast_format=None,
        optional_arg=True,
        type_expr=None,
    ):
        """
        optional_arg: controlling whether None will be returned instead of a Node. e.g. optional_arg = linear_mod.bias will return None when the module does not have a bias term.
                      Defaults to True.
        """
        if optional_arg is None:
            return None
        node = super().get_attr(qualified_name, type_expr)
        node = self.qdq_node(node, cast_name, cast_format)
        return node

    def create_node(
        self,
        op,
        target,
        args=None,
        kwargs=None,
        name=None,
        type_expr=None,
        cast_name=None,
        cast_format=None,
    ):
        node = super().create_node(op, target, args, kwargs, name, type_expr)
        node = self.qdq_node(node, cast_name, cast_format)
        return node

    def call_function(
        self,
        the_function,
        args=None,
        kwargs=None,
        type_expr=None,
        cast_name=None,
        cast_format=None,
    ):
        node = super().call_function(the_function, args, kwargs, type_expr)
        node = self.qdq_node(node, cast_name, cast_format)
        return node

    def call_method(
        self,
        method_name,
        args=None,
        kwargs=None,
        type_expr=None,
        cast_name=None,
        cast_format=None,
    ):
        node = super().call_method(method_name, args, kwargs, type_expr)
        node = self.qdq_node(node, cast_name, cast_format)
        return node
