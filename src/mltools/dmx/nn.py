from functools import reduce
from typing import Union, List, Tuple, Optional
from collections import OrderedDict
import sys
from types import SimpleNamespace
import torch
from torch import Tensor, Size
import torch.nn.functional as F
import transformers
import math

from mltools.numerical import (
    Format,
    NumericalCastMixin,
    Same,
    FixedPoint,
    FloatingPoint,
    BlockFloatingPoint,
    CastTo,
)
from mltools.sparse import (
    Sparseness,
    WeightSparseMixin,
    Dense,
    TopK,
    BlockTopK,
    Bernoulli,
    Sparsify,
    LazySparsify,
)
from mltools.functional import (
    ApproximationFunction,
    ApproximationMixin,
    NoApproximation,
    SoftmaxApproximation,
    GELUApproximation,
    LayerNormApproximation,
    Approximate,
)
from mltools.perf_proxy import PerformanceProxyMixin
from mltools.layer_reconstruction import LayerReconstructionMixin


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

    def __init__(self, *args, state_dict_url: Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state_dict_url = state_dict_url
        if state_dict_url is not None:
            self.load_state_dict_and_register_url(state_dict_url)

    def transform(self, config) -> None:
        """
        A function that changes the format of the ops and loading state dicts according to the config file.

        Args:
            config (DmxModuleConfig): config file for setting new formats and loading state dicts.

        """
        # numerics transformation
        if "input_format" in config:
            self.input_cast.set_format(format=config["input_format"])
        if "output_format" in config:
            self.output_cast.set_format(format=config["output_format"])
        if self.residual_cast is not None and "residual_format" in config:
            self.residual_cast.set_format(format=config["residual_format"])
        if self.multiplier_cast is not None and "multiplier_format" in config:
            self.multiplier_cast.set_format(format=config["multiplier_format"])
        if self.accum_cast is not None and "accum_format" in config:
            self.accum_cast.set_format(format=config["accum_format"])
        if self.weight_cast is not None and "weight_format" in config:
            self.weight_cast.set_format(format=config["weight_format"])
        if self.bias_cast is not None and "bias_format" in config:
            self.bias_cast.set_format(format=config["bias_format"])
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
        from mltools.utils import save_state_dict_and_return_url

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
        if self.smoothquant is not None:
            if self.smoothquant.dynamic[0] == 1 or self.smoothquant.calibrating:
                self.update_smoothquant_scale(input)
            input = self.smoothquant.scale_input(input)
        _input = self.input_cast(input)
        if self.obc is not None:
            self.obc.measure_hessian(_input)
        _output = self._forward(_input, *args, **kwags)
        output = self.output_cast(_output)
        if self.flop_counter_enabled:
            self.count_flops(input, output)
        return output.to(_dtype).to(_device)

    def update_params_with_raw(self, raw: torch.nn.Module) -> None:
        """
        Update parameters of a DmxModule from a torch.nn.Module.

        Args:
            raw (torch.nn.Module): the torch module to copy parameters from.
        """
        state_dic = self.state_dict()
        for key, val in raw.state_dict().items():
            state_dic[key] = val
        self.load_state_dict(state_dic)
        # inherit some module attributes from raw module
        self.training = raw.training
        if hasattr(raw, "dtype"):
            self.dtype = raw.dtype


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
        cc.instance = module.__class__
        if isinstance(module, DmxModule):
            if module.input_format is not None and (
                freeze or module.input_format != "SAME"
            ):
                cc.input_format = module.input_format
            if module.residual_format is not None and (
                freeze or module.residual_format != "SAME"
            ):
                cc.residual_format = module.residual_format
            if module.multiplier_format is not None and (
                freeze or module.multiplier_format != "SAME"
            ):
                cc.multiplier_format = module.multiplier_format
            if module.output_format is not None and (
                freeze or module.output_format != "SAME"
            ):
                cc.output_format = module.output_format
            if module.accum_format is not None and (
                freeze or module.accum_format != "SAME"
            ):
                cc.accum_format = module.accum_format
            if module.weight_format is not None and (
                freeze or module.weight_format != "SAME"
            ):
                cc.weight_format = module.weight_format
            if module.bias_format is not None and (
                freeze or module.bias_format != "SAME"
            ):
                cc.bias_format = module.bias_format
            if module.weight_sparseness is not None and (
                freeze or module.weight_sparseness != "DENSE"
            ):
                cc.weight_sparseness = module.weight_sparseness
            if freeze or module.approximation_function != "NONE":
                cc.approximation_function = module.approximation_function
            if module.state_dict_url is not None:
                cc.state_dict_url = module.state_dict_url
        return cc.__dict__


is_configurable = lambda m: isinstance(m, DmxModule)


class ResAdd(DmxModule, torch.nn.Module):
    """
    A module for handling residual connections.

    Attributes:
        residual_cast (CastTo): CastTo module for residual component
    """

    def __init__(self) -> None:
        super().__init__()
        self.residual_cast = CastTo()

    def _forward(self, _input: Tensor, residual: Tensor) -> Tensor:
        """
        A forward pass of addition operation with quantization applied

        Args:
            _input (Tensor): already quantized input tensor
            residual (Tensor): residual tensor

        Returns:
            Sum of _input tensor and quantized residual tensor.
        """
        _residual = self.residual_cast(residual)
        _output = _input + _residual
        return _output


class ActActMatMul(DmxModule, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.multiplier_cast = CastTo()

    def _forward(self, _input: Tensor, multiplier: Tensor) -> Tensor:
        _multiplier = self.multiplier_cast(multiplier)
        _output = torch.matmul(_input, _multiplier)
        return _output


class Linear(DmxModule, torch.nn.Linear):
    r"""
    An extension of PyTorch's Linear layer to support DmxModule configurations.
    This module performs a linear transformation on the input data.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
        **kwargs: Additional keyword arguments inherited from DmxModule.

    Attributes:
        _weight (Tensor): The learnable weights of the module of shape (out_features, in_features).
        _bias (Tensor): The learnable bias of the module of shape (out_features).

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, **kwargs)

    def _forward(self, _input: Tensor) -> Tensor:
        if self.accum_format == "SAME":
            _weight = self._weight.to(_input.dtype)
            _bias = None if self._bias is None else self._bias.to(_input.dtype)
            _output = torch.nn.functional.linear(_input, _weight, _bias)
        else:
            _weight = self._weight
            _product = self.accum_cast(
                torch.matmul(_input.to(_weight.dtype), _weight.t())
            )
            if self.bias is not None:
                _output = torch.add(_product, self._bias)
            else:
                _output = _product

        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        initial_dmx = Linear(raw.in_features, raw.out_features, bias=raw.bias != None)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class HFTransformersConv1D(DmxModule, transformers.pytorch_utils.Conv1D):
    def __init__(self, nf: int, nx: int) -> None:
        super().__init__(nf, nx)

    def _forward(self, _input: Tensor) -> Tensor:
        size_out = _input.size()[:-1] + (self.nf,)
        _output = torch.addmm(
            self._bias, _input.view(-1, _input.size(-1)), self._weight
        )
        _output = _output.view(size_out)
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        initial_dmx = HFTransformersConv1D(raw.weight.shape[1], raw.weight.shape[0])
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class Conv1d(DmxModule, torch.nn.Conv1d):
    r"""
    An extension of PyTorch's Conv1d layer to support DmxModule configurations.
    This module performs a 1D convolution over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
        padding_mode (str, optional): Accepted values 'zeros' and 'circular' etc. Defaults to 'zeros'.
        **kwargs: Additional keyword arguments inherited from DmxModule.

    Attributes:
        _weight (Tensor): The learnable weights of the module of shape (out_channels, in_channels, kernel_size).
        _bias (Tensor, optional): The learnable bias of the module of shape (out_channels).

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the 1D convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            **kwargs,
        )

    def _forward(self, _input: Tensor) -> Tensor:
        _weight = self._weight
        _convolution = self.accum_cast(
            self._conv_forward(_input.to(_weight.dtype), _weight, None)
        )
        if self.bias is not None:
            _output = torch.add(_convolution, self._bias.unsqueeze(-1).unsqueeze(-1))
        else:
            _output = _convolution
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new Conv1d object (DmxModule) from a given PyTorch Conv1d layer.

        Args:
            raw (torch.nn.Module): A PyTorch Conv1d layer to be converted.

        Returns:
            DmxModule: A Conv1d object that has the same configuration as the input PyTorch Conv1d layer.
        """
        initial_dmx = Conv1d(
            raw.in_channels,
            raw.out_channels,
            raw.kernel_size,
            stride=raw.stride,
            padding=raw.padding,
            dilation=raw.dilation,
            groups=raw.groups,
            bias=raw.bias != None,
            padding_mode=raw.padding_mode,
        )
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class Conv2d(DmxModule, torch.nn.Conv2d):
    r"""
    An extension of PyTorch's Conv2d layer to support DmxModule configurations.
    This module performs a 2D convolution over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
        padding_mode (str, optional): Accepted values 'zeros' and 'circular' etc. Defaults to 'zeros'.
        **kwargs: Additional keyword arguments inherited from DmxModule.

    Attributes:
        _weight (Tensor): The learnable weights of the module of shape (out_channels, in_channels, kernel_height, kernel_width).
        _bias (Tensor, optional): The learnable bias of the module of shape (out_channels).

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the 2D convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            **kwargs,
        )

    def _forward(self, _input: Tensor) -> Tensor:
        _weight = self._weight
        _convolution = self.accum_cast(
            self._conv_forward(_input.to(_weight.dtype), _weight, None)
        )
        if self.bias is not None:
            _output = torch.add(_convolution, self._bias.unsqueeze(-1).unsqueeze(-1))
        else:
            _output = _convolution
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new Conv2d object (DmxModule) from a given PyTorch Conv2d layer.

        Args:
            raw (torch.nn.Module): A PyTorch Conv2d layer to be converted.

        Returns:
            DmxModule: A Conv2d object that has the same configuration as the input PyTorch Conv2d layer.
        """
        initial_dmx = Conv2d(
            raw.in_channels,
            raw.out_channels,
            raw.kernel_size,
            stride=raw.stride,
            padding=raw.padding,
            dilation=raw.dilation,
            groups=raw.groups,
            bias=raw.bias != None,
            padding_mode=raw.padding_mode,
        )
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class ConvTranspose2d(DmxModule, torch.nn.ConvTranspose2d):
    r"""
    An extension of PyTorch's ConvTranspose2d layer to support DmxModule configurations.
    This module performs a 2D transposed convolution over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the transposed convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the transposed convolution. Defaults to 1.
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
        padding_mode (str, optional): Accepted values 'zeros'. Defaults to 'zeros'.
        **kwargs: Additional keyword arguments inherited from DmxModule.

    Attributes:
        _weight (Tensor): The learnable weights of the module of shape (in_channels, out_channels, kernel_height, kernel_width).
        _bias (Tensor, optional): The learnable bias of the module of shape (out_channels).

    Methods:
        _forward (_input: Tensor, output_size: Optional[List[int]] = None) -> Tensor: Computes the forward pass of the 2D transposed convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        padding_mode="zeros",
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            **kwargs,
        )

    def _forward(
        self, _input: Tensor, output_size: Optional[List[int]] = None
    ) -> Tensor:
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 2
        output_padding = self._output_padding(
            _input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,
        )  # type: ignore[arg-type]

        _weight = self._weight
        _convolution = self.accum_cast(
            F.conv_transpose2d(
                _input.to(_weight.dtype),
                _weight,
                None,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
        )
        if self.bias is not None:
            _output = torch.add(_convolution, self._bias.unsqueeze(-1).unsqueeze(-1))
        else:
            _output = _convolution
        return _output


class AdaptiveAvgPool2d(DmxModule, torch.nn.AdaptiveAvgPool2d):
    r"""
    An extension of PyTorch's AdaptiveAvgPool2d layer to support DmxModule configurations.
    This module applies a 2D adaptive average pooling over an input signal composed of several input planes.

    Args:
        output_size (int or tuple): The size of the output tensor after pooling.

    Attributes:
        None specific to this subclass. Inherits attributes from parent classes.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the 2D adaptive average pooling.
    """

    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new AdaptiveAvgPool2d object (DmxModule) from a given PyTorch AdaptiveAvgPool2d layer.

        Args:
            raw (torch.nn.Module): A PyTorch AdaptiveAvgPool2d layer to be converted.

        Returns:
            DmxModule: An AdaptiveAvgPool2d object that has the same configuration as the input PyTorch AdaptiveAvgPool2d layer.
        """
        initial_dmx = AdaptiveAvgPool2d(raw.output_size)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class AvgPool2d(DmxModule, torch.nn.AvgPool2d):
    r"""
    An extension of PyTorch's AvgPool2d layer to support DmxModule configurations.
    This module applies a 2D average pooling over an input signal composed of several input planes.

    Args:
        output_size (int or tuple): The size of the output tensor after pooling.

    Attributes:
        None specific to this subclass. Inherits attributes from parent classes.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the 2D average pooling.
    """

    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new AvgPool2d object (DmxModule) from a given PyTorch AvgPool2d layer.

        Args:
            raw (torch.nn.Module): A PyTorch AvgPool2d layer to be converted.

        Returns:
            DmxModule: An AvgPool2d object that has the same configuration as the input PyTorch AvgPool2d layer.
        """
        initial_dmx = AvgPool2d(raw.output_size)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class MaxPool2d(DmxModule, torch.nn.MaxPool2d):
    r"""
    An extension of PyTorch's MaxPool2d layer to support DmxModule configurations.
    This module applies a 2D max pooling over an input signal composed of several input planes.

    Args:
        kernel_size (int or tuple): Size of the window to take a max over.
        stride (int or tuple, optional): Stride of the window. Defaults to None.
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        return_indices (bool, optional): If True, will return the max indices in a second tensor. Defaults to False.
        ceil_mode (bool, optional): If True, will use ceil instead of floor to compute the output shape. Defaults to False.

    Attributes:
        None specific to this subclass. Inherits attributes from parent classes.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the 2D max pooling.
    """

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ) -> None:
        super().__init__(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new MaxPool2d object (DmxModule) from a given PyTorch MaxPool2d layer.

        Args:
            raw (torch.nn.Module): A PyTorch MaxPool2d layer to be converted.

        Returns:
            DmxModule: A MaxPool2d object that has the same configuration as the input PyTorch MaxPool2d layer.
        """
        initial_dmx = MaxPool2d(
            raw.kernel_size,
            stride=raw.stride,
            padding=raw.padding,
            dilation=raw.dilation,
            return_indices=raw.return_indices,
            ceil_mode=raw.ceil_mode,
        )
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class Softmax(DmxModule, torch.nn.Softmax):
    r"""
    An extension of PyTorch's Softmax layer to support DmxModule configurations.
    This module applies the Softmax function to an n-dimensional input tensor, normalizing the elements
    along a specified dimension such that they sum up to 1.

    Args:
        dim (int, optional): Dimension along which Softmax will be computed. Defaults to -1.

    Attributes:
        None specific to this subclass. Inherits attributes from parent classes.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the Softmax function.
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__(dim=dim)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,), dim=self.dim)
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        """
        Creates a Softmax DmxModule instance from a raw torch.nn.Module instance.

        Args:
            raw (torch.nn.Module): The raw torch.nn.Module instance.

        Returns:
            DmxModule: An initialized Softmax DmxModule instance with parameters copied from the raw instance.
        """
        initial_dmx = Softmax(dim=raw.dim)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class LayerNorm(DmxModule, torch.nn.LayerNorm):
    r"""
    An extension of PyTorch's LayerNorm layer to support DmxModule configurations.
    This module applies layer normalization over dimensions specified by the `normalized_shape` attribute.
    The mean and standard deviation are computed over the last `D` dimensions, where `D`is the dimensionality indicated by `normalized_shape`.
    Gamma and Beta are learnable parameters if `elementwise_affine` is set to True.

    Args:
        normalized_shape (Union[int, List[int], Size]): Specifies dimensions for layer normalization.
        eps (float, optional): A value added for numerical stability. Defaults to 1e-5.
        elementwise_affine (bool, optional): Indicates if learnable affine parameters Gamma and Beta should be used. Defaults to True.

    Attributes:
        None specific to this subclass. Inherits attributes from parent classes.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the layer normalization.
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine
        )

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward(
            (_input,), self.normalized_shape, self._weight, self._bias, self.eps
        )
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        """
        Creates a new LayerNorm object (DmxModule) from a given PyTorch LayerNorm layer.

        Args:
            raw (torch.nn.Module): A PyTorch LayerNorm layer to be converted.

        Returns:
            DmxModule: A LayerNorm object that has the same configuration as the input PyTorch LayerNorm layer.
        """
        initial_dmx = LayerNorm(
            raw.normalized_shape, eps=raw.eps, elementwise_affine=raw.elementwise_affine
        )
        initial_dmx.update_params_with_raw(raw)
        initial_dmx.type(raw.weight.dtype)
        return initial_dmx


class HFTransformersT5LayerNorm(
    DmxModule, transformers.models.t5.modeling_t5.T5LayerNorm
):
    r"""
    An extension of the Hugging Face Transformers T5 LayerNorm layer to support DmxModule configurations.
    This module applies RMS-based layer normalization over the input tensor.
    The layer normalization is parameterized by the `hidden_size` and optional `eps` value for numerical stability.

    Args:
        hidden_size (int): The size of the hidden layer (number of hidden units).
        eps (float, optional): A small constant added to the denominator for numerical stability. Defaults to 1e-6.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the layer normalization.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(hidden_size, eps=eps)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output


class HFTransformersLlamaRMSNorm(
    DmxModule, transformers.models.llama.modeling_llama.LlamaRMSNorm
):
    r"""
    An extension of the Hugging Face Transformers Llama RMSNorm layer to support DmxModule configurations.
    This module performs RMS-based layer normalization on the input tensor.
    The layer normalization is characterized by the `hidden_size` and an optional `eps` value for numerical stability.

    Args:
        hidden_size (int): The size of the hidden layer (number of hidden units).
        eps (float, optional): A small constant added to the denominator for numerical stability. Defaults to 1e-6.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the RMS layer normalization.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(hidden_size, eps=eps)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output


class BatchNorm2d(DmxModule, torch.nn.BatchNorm2d):
    r"""
    An extension of PyTorch's BatchNorm2d layer to support DmxModule configurations.
    This module applies batch normalization over a 4D input tensor, suitable for use with 2D convolutional layers.
    The module is parameterized by the number of features, epsilon value for numerical stability, momentum for the running mean and variance, and options to use affine transformation and track running statistics.

    Args:
        num_features (int): Number of channels in the input tensor.
        eps (float, optional): A small constant added to the denominator for numerical stability. Defaults to 1e-05.
        momentum (float, optional): The momentum value for the running mean and running variance computation. Defaults to 0.1.
        affine (bool, optional): Whether to include learnable affine parameters for this layer. Defaults to True.
        track_running_stats (bool, optional): Whether to track the running mean and variance during training. Defaults to True.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the batch normalization.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def _forward(self, _input: Tensor) -> Tensor:
        self._check_input_dim(_input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        _output = F.batch_norm(
            _input,
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self._weight,
            self._bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new BatchNorm2d object (DmxModule) from a given PyTorch BatchNorm2d layer.

        Args:
            raw (torch.nn.Module): A PyTorch BatchNorm2d layer to be converted.

        Returns:
            DmxModule: A BatchNorm2d object that has the same configuration as the input PyTorch BatchNorm2d layer.
        """
        initial_dmx = BatchNorm2d(
            raw.num_features,
            eps=raw.eps,
            momentum=raw.momentum,
            affine=raw.affine,
            track_running_stats=raw.track_running_stats,
        )
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class GroupNorm(DmxModule, torch.nn.GroupNorm):
    r"""
    An extension of PyTorch's GroupNorm layer to support DmxModule configurations.
    This module applies group normalization over an input tensor, suitable for use with various types of layers.
    The module is parameterized by the number of groups, number of channels, epsilon value for numerical stability, and an option to use affine transformation.

    Args:
        num_groups (int): Number of groups to separate the channels into.
        channels (int): Number of channels in the input tensor.
        eps (float, optional): A small constant added to the denominator for numerical stability. Defaults to 1e-5.
        affine (bool, optional): Whether to include learnable affine parameters for this layer. Defaults to True.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the group normalization.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)

    def _forward(self, input: Tensor) -> Tensor:
        _weight = self._weight
        _bias = self._bias
        _input = input.to(_weight.dtype) if _weight is not None else input

        _output = F.group_norm(_input, self.num_groups, _weight, _bias, self.eps)

        return _output


class Dropout(DmxModule, torch.nn.Dropout):
    r"""
    An extension of PyTorch's Dropout layer to support DmxModule configurations.
    This module applies the dropout operation over the input tensor.

    Args:
        p (float, optional): The probability of an element to be zeroed. Defaults to 0.5.
        inplace (bool, optional): If set to ``True``, will do this operation in-place. Defaults to False.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the dropout layer.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p=p, inplace=inplace)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new Dropout object (DmxModule) from a given PyTorch Dropout layer.

        Args:
            raw (torch.nn.Module): A PyTorch Dropout layer to be converted.

        Returns:
            DmxModule: A Dropout object that has the same configuration as the input PyTorch Dropout layer.
        """
        initial_dmx = Dropout(p=raw.p, inplace=raw.inplace)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class ReLU(DmxModule, torch.nn.ReLU):
    r"""
    An extension of PyTorch's ReLU layer to support DmxModule configurations.
    This module applies the Rectified Linear Unit (ReLU) function element-wise on the input tensor.

    Args:
        inplace (bool, optional): If set to ``True``, will do this operation in-place. Defaults to False.

    Methods:
        _forward (_input: Tensor, inplace: bool = False) -> Tensor: Computes the forward pass of the ReLU layer.
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)

    def _forward(self, _input: Tensor, inplace: bool = False) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new ReLU object (DmxModule) from a given PyTorch ReLU layer.

        Args:
            raw (torch.nn.Module): A PyTorch ReLU layer to be converted.

        Returns:
            DmxModule: A ReLU object that has the same configuration as the input PyTorch ReLU layer.
        """
        initial_dmx = ReLU(inplace=raw.inplace)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class ReLU6(DmxModule, torch.nn.ReLU6):
    r"""
    An extension of PyTorch's ReLU6 layer to support DmxModule configurations.
    This module applies the Rectified Linear Unit 6 (ReLU6) function element-wise on the input tensor.

    Args:
        inplace (bool, optional): If set to ``True``, will do this operation in-place. Defaults to False.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the ReLU6 layer.
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new ReLU6 object (DmxModule) from a given PyTorch ReLU6 layer.

        Args:
            raw (torch.nn.Module): A PyTorch ReLU6 layer to be converted.
        Returns:
            DmxModule: A ReLU6 object that has the same configuration as the input PyTorch ReLU6 layer.
        """
        initial_dmx = ReLU6(inplace=raw.inplace)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class SiLU(DmxModule, torch.nn.SiLU):
    r"""
    An extension of PyTorch's SiLU (Sigmoid Linear Unit) layer to support DmxModule configurations.
    This module applies the SiLU function element-wise on the input tensor.

    Args:
        inplace (bool, optional): If set to ``True``, will do this operation in-place. Defaults to False.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the SiLU layer.
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new SiLU object (DmxModule) from a given PyTorch SiLU layer.

        Args:
            raw (torch.nn.Module): A PyTorch SiLU layer to be converted.

        Returns:
            DmxModule: A SiLU object that has the same configuration as the input PyTorch SiLU layer.
        """
        initial_dmx = SiLU(inplace=raw.inplace)
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class Tanh(DmxModule, torch.nn.Tanh):
    r"""
    An extension of PyTorch's Tanh (Hyperbolic Tangent) layer to support DmxModule configurations.
    This module applies the tanh function element-wise on the input tensor.

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the Tanh layer.
    """

    def __init__(self) -> None:
        super().__init__()

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new Tanh object (DmxModule) from a given PyTorch Tanh layer.

        Args:
            raw (torch.nn.Module): A PyTorch Tanh layer to be converted.

        Returns:
            DmxModule: A Tanh object that has the same configuration as the input PyTorch Tanh layer.
        """
        initial_dmx = Tanh()
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx


class GELU(DmxModule, torch.nn.GELU):
    r"""
    An extension of PyTorch's GELU (Gaussian Error Linear Unit) layer to support DmxModule configurations.
    This module applies the GELU function element-wise on the input tensor.
    The function is defined as:

    Methods:
        _forward (_input: Tensor) -> Tensor: Computes the forward pass of the GELU layer.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _forward(self, _input: Tensor) -> Tensor:
        _output = self.approx_forward((_input,))
        return _output

    @staticmethod
    def from_raw(raw: torch.nn.Module) -> DmxModule:
        r"""
        Creates a new GELU object (DmxModule) from a given PyTorch GELU layer.

        Args:
            raw (torch.nn.Module): A PyTorch GELU layer to be converted.

        Returns:
            DmxModule: A GELU object that has the same configuration as the input PyTorch GELU layer.
        """
        initial_dmx = GELU()
        initial_dmx.update_params_with_raw(raw)
        return initial_dmx
