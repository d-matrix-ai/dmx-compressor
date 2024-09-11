from typing import Union
import torch
import torch.nn as nn
from .format import Format


class SmoothQuant(nn.Module):
    """
    SmoothQuant is a quantization technique that reduces MatMul quantization error
    by migrating the quantization difficulty from the first input of the MatMul
    (input A) to the second input (input B).

    .. smoothQuant paper:
    https://arxiv.org/pdf/2211.10438.pdf

    Args:
        `a_ch_axis` (int): channel axis for input A of the MatMul
        `b_ch_axis` (int): channel axis for input B of the MatMul
        `a_dynamic` (bool): If set to True, the maximum value of input A will be
            calculated dynamically, default is False.
        `b_dynamic` (bool): If set to True, the maximum value of input B will be
            calculated dynamically, default is False.
        `migration_strength` (float): controls how much quantization difficulty
            we want to migrate from input A to input B, should be between 0 and 1,
            default is 0.5.
        `scale_format` (str or dmx.Format): the numerical format to
            store and compute the scaler, default is "SAME".
        `scale_min`(float): minimum epsilon value used to prevent division by zero
            calculating the scaling factors, default is 1e-5.

    Attributes:
        `a_ch_axis` (int): channel axis for input A of the MatMul
        `b_ch_axis` (int): channel axis for input B of the MatMul
        `a_dynamic` (bool): If set to True, the maximum value of input A will be
            calculated dynamically, default is False.
        `b_dynamic` (bool): If set to True, the maximum value of input B will be
            calculated dynamically, default is False.
        `migration_strength` (float): controls how much quantization difficulty
            we want to migrate from input A to input B, should be between 0 and 1,
            default is 0.5.
        `scale_format` (str or dmx.Format): the numerical format to
            store and compute the scaler, default is "SAME".
        `scale_min` (float): minimum epsilon value used to prevent division by zero
            calculating the scaling factors, default is 1e-5.
        `enabled` (bool): If set to True, smoothQuant will be enabled for both
            input A and input B
        `scale` (Tensor): scaling factors used to scale input A and input B to
            (input A / scale) and (input B * scale), respectively.
        `a_maxabs` (Tensor): the maximum value of absolute of input A
        `b_maxabs` (Tensor): the maximum value of absolute of input B
    """

    calibrating: bool = False

    def __init__(
        self,
        a_ch_axis: int,
        b_ch_axis: int,
        a_dynamic: bool = False,
        b_dynamic: bool = False,
        migration_strength: float = 0.5,
        scale_format: Union[str, Format] = "SAME",
        scale_min: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.a_ch_axis = a_ch_axis
        self.b_ch_axis = b_ch_axis
        self.register_buffer(
            "a_dynamic", torch.tensor([int(a_dynamic)], dtype=torch.long)
        )
        self.register_buffer(
            "b_dynamic", torch.tensor([int(b_dynamic)], dtype=torch.long)
        )
        self.register_buffer("enabled", torch.tensor([0], dtype=torch.long))
        self.register_buffer("migration_strength", torch.tensor([migration_strength]))
        self.register_buffer("scale_min", torch.tensor([scale_min]))
        self.register_buffer("scale", torch.empty(0))
        self.register_buffer("a_maxabs", torch.empty(0), persistent=False)
        self.register_buffer("b_maxabs", torch.empty(0))

        from .cast import CastTo

        self.scale_cast = CastTo()
        self.set_scale_format(scale_format)

    @torch.jit.export
    def enable(self, enabled: bool = True) -> None:
        """
        Sets/resets the enabled flag.

        Args:
            enabled (bool): if set to True, smoothQuant is enabled, default is True.
        """
        self.enabled[0] = 1 if enabled else 0

    @torch.jit.export
    def disable(self) -> None:
        """
        Disables smoothQuant.
        """
        self.enable(False)

    @torch.jit.export
    def set_dynamic(self, a_dynamic: bool = True, b_dynamic: bool = True) -> None:
        """
        Sets/resets the dynamic flag for inputs A and B.

        Args:
            a_dynamic (bool): if set to True, the maximum value of input A will be
                calculated dynamically, default is True.
            b_dynamic (bool): if set to True, the maximum value of input B will be
                calculated dynamically, default is True.
        """
        self.a_dynamic[0] = 1 if a_dynamic else 0
        self.b_dynamic[0] = 1 if b_dynamic else 0

    @torch.jit.export
    def set_scale_format(self, format: Union[str, Format] = "SAME") -> None:
        """
        Sets/resets the scale_format.

        Args:
            `format` (str or dmx.Format): the numerical format to
        store and compute the scaler, default is "SAME".
        """
        self.scale_cast.set_format(format)

    @torch.jit.export
    def set_migration_strength(self, migration_strength: float) -> None:
        """
        Sets the migration_strength factor.

        Args:
            migration_strength (float): quantization difficulty migration factor,
                should be between 0 and 1, default is 0.5.

        Raises:
            ValueError: If migration_strength is less than 0.0 or greater than 1.0.
        """
        if 0.0 <= migration_strength <= 1.0:
            self.migration_strength[0] = migration_strength
        else:
            raise ValueError(
                f"migration_strength should be between 0 and 1, got {migration_strength}"
            )

    @torch.jit.export
    def reset_scale(self) -> None:
        """
        Resets the scaling tensor to an empty tensor.
        """
        self.scale.data = torch.empty(0)

    @torch.jit.export
    def reset_a_maxabs(self) -> None:
        """
        Resets a_maxabs to an empty tensor.
        """
        self.a_maxabs.data = torch.empty(0)

    @torch.jit.export
    def reset_b_maxabs(self) -> None:
        """
        Resets b_maxabs to an empty tensor.
        """
        self.b_maxabs.data = torch.empty(0)

    @property
    def a_maxabs_exists(self) -> bool:
        """
        Checks if a_maxabs is already calculated.

        Returns:
            True if a_maxabs is calculated, False otherwise.
        """
        return self.a_maxabs.numel() > 0

    @property
    def b_maxabs_exists(self) -> bool:
        """
        Checks if b_maxabs is already calculated.

        Returns:
            True if b_maxabs is calculated, False otherwise.
        """
        return self.b_maxabs.numel() > 0

    def _slicing(self, x: torch.Tensor, dims: torch.Size) -> torch.Tensor:
        """
        Slices the input tensor x such that the size of the sliced output tensor
            matches the given dimensions.

        Args:
            x (Tensor): the input tensor
            dims (Size): the target dimensions

        Returns:
            the output tensor

        Raises:
            RuntimeError: If the dimensions of the input tensor do not match the slicing dimensions.
        """
        if len(x.size()) != len(dims):
            raise RuntimeError(
                "Input tensor should have the same number of dimensions as slicing dimensions"
            )
        slice_str = "["
        for dim in dims:
            slice_str += f"0:{dim},"
        slice_str = slice_str[:-1] + "]"
        return eval(f"x{slice_str}")

    def _padding(self, x: torch.Tensor, dims: torch.Size) -> torch.Tensor:
        """
        Pads the input tensor x such that the size of the padded output tensor
            matches the given dimensions.

        Args:
            x (Tensor): the input tensor
            dims (Size): the target dimensions

        Returns:
            the output tensor

        Raises:
            RuntimeError: If the dimensions of the input tensor do not match the padding dimensions.
        """
        _x_dims = x.size()
        if len(_x_dims) != len(dims):
            raise RuntimeError(
                "Input tensor should have the same number of dimensions as padding dimensions"
            )
        pad_nd_str = "("
        for x_dim, dim in zip(reversed(_x_dims), reversed(dims)):
            pad_nd_str += f"0,{dim - x_dim},"
        pad_nd_str = pad_nd_str[:-1] + ")"
        return nn.functional.pad(x, eval(f"{pad_nd_str}"), "constant", 0)

    def _proper_shape(self, x: torch.Tensor, dim: int) -> torch.Size:
        """
        Calculates the proper shape for the scaling factor of a given tensor.

        Args:
            x: input tensor
            dim: the dimension that scaling should be applied on

        Returns:
            shape of the scaling tensor
        """
        sz = [1] * x.dim()
        sz[dim] = self.scale.numel()
        return torch.Size(sz)

    def scale_a(self, a: torch.Tensor) -> torch.Tensor:
        """
        If smoothQuant is enabled, scales input A.

        Args:
            a (Tensor): input tensor that scaling will be applied on

        Returns:
            scaled input tensor
        """
        if self.enabled[0] == 1:
            sz = self._proper_shape(a, self.a_ch_axis)
            a = a.to(self.scale.device) / self.scale.view(sz)
        return a

    def scale_b(self, b: torch.Tensor) -> torch.Tensor:
        """
        If smoothQuant is enabled, scales input B.

        Args:
            b (Tensor): input tensor that scaling will be applied on

        Returns:
            scaled input tensor
        """
        if self.enabled[0] == 1:
            sz = self._proper_shape(b, self.b_ch_axis)
            b = b.to(self.scale.device) * self.scale.view(sz)
        return b

    def _maxabs(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Removes the given dim from the input tensor dimension and computes the maximum
        value of each slice of the input tensor in the remaining dimension(s).

        Args:
            x (Tensor): the input tensor
            dim (int): dimension that will be excluded from computing max

        Returns:
            the maximum value of each slice of the input tensor in all dimensions, except the given dim.
        """
        dims = list(range(x.dim()))
        dims.pop(dim)
        return torch.amax(x.abs(), dim=dims)

    def compute_scale(self, a_maxabs: torch.Tensor, b_maxabs: torch.Tensor) -> None:
        """
        Computes the scaling tensor.

        Args:
            a_maxabs (Tensor): the maximum value of absolute of input A
            b_maxabs (Tensor): the maximum value of absolute of input B
        """
        b_maxabs = b_maxabs.to(self.scale_min.device).clamp(
            min=self.scale_min
        )  # to prevent division-by-zero
        _device = self.migration_strength.device
        _scale = (
            (
                (a_maxabs.to(_device) ** self.migration_strength)
                / (b_maxabs.to(_device) ** (1.0 - self.migration_strength))
            )
            .to(self.scale_min.device)
            .clamp(min=self.scale_min)
        )
        self.scale = self.scale_cast(_scale)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> None:
        """
        Computes the smoothQuant scaling tensor and scales inputs A and B

        Args:
            a (tensor): input tensor A
            b (tensor): input tensor B
        """
        with torch.no_grad():
            if not self.a_maxabs_exists or self.a_dynamic[0] == 1:
                self.a_maxabs = self._maxabs(a, self.a_ch_axis)
            else:
                self.a_maxabs = torch.maximum(
                    self._padding(
                        self._maxabs(a, self.a_ch_axis), self.a_maxabs.size()
                    ),
                    self.a_maxabs,
                )
            if not self.b_maxabs_exists or self.b_dynamic[0] == 1:
                self.b_maxabs = self._maxabs(b, self.b_ch_axis)
            else:
                self.b_maxabs = torch.maximum(
                    self._padding(
                        self._maxabs(b, self.b_ch_axis), self.b_maxabs.size()
                    ),
                    self.b_maxabs,
                )
            self.compute_scale(
                self._slicing(self.a_maxabs, self._maxabs(a, self.a_ch_axis).size()),
                self._slicing(self.b_maxabs, self._maxabs(b, self.b_ch_axis).size()),
            )
            return self.scale_a(a), self.scale_b(b)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """
        Overrides torch.nn.Module._load_from_state_dict() to avoid tensor shape mismatch
        """
        if prefix + "scale" in state_dict.keys():
            self.scale = state_dict[prefix + "scale"]  # asign scale manually
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def extra_repr(self) -> str:
        """
        Returns the extra representation of smoothQuant
        """
        return f"migration_strength = {self.migration_strength.item()}, a_ch_axis = {self.a_ch_axis}, b_ch_axis = {self.b_ch_axis}, scale_format = {self.scale_cast.format}, dynamic = ({self.a_dynamic.bool().item()}, {self.b_dynamic.bool().item()})"


class ActivationWeightSmoothQuant(SmoothQuant):
    """
    This is the derived class for Activation x Weight smoothQuant.

    Args:
        `ch_axis` (int): channel axis for the input activation tensor
        `win_ch_axis` (int): channel axis for the weight tensor
        `migration_strength` (float): controls how much quantization difficulty
            we want to migrate from activations to weights, should be between
            0 and 1, default is 0.5.
        `scale_format` (str or dmx.Format): the numerical format to
            store and compute the scaler, default is "SAME".
        `dynamic` (bool): If set to True, the maximum value of activations will
            be calculated dynamically, default is False.
        `scale_min`(float): minimum epsilon value used to prevent division by
            zero calculating the scaling factors, default is 1e-5.

    Attributes:
        `ch_axis` (int): channel axis for the input activation tensor
        `win_ch_axis` (int): channel axis for the weight tensor
        `fused_to_weight` (bool): If set to True, the scaling factors will be
            fused to the weights, cannot be enabled when dynamic is set.
    """

    def __init__(
        self,
        ch_axis: int,
        win_ch_axis: int,
        migration_strength: float = 0.5,
        scale_format: Union[str, Format] = "SAME",
        dynamic: bool = False,
        scale_min: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__(
            a_ch_axis=ch_axis,
            b_ch_axis=win_ch_axis,
            migration_strength=migration_strength,
            scale_format=scale_format,
            a_dynamic=dynamic,
            b_dynamic=False,
            scale_min=scale_min,
            **kwargs,
        )
        self.ch_axis = ch_axis
        self.win_ch_axis = win_ch_axis
        self.register_buffer("fused_to_weight", torch.tensor([0], dtype=torch.long))

    @torch.jit.export
    def set_dynamic(self, dynamic: bool = True) -> None:
        """
        Sets/resets the dynamic flag for the input activation

        Args:
            a_dynamic (bool): if set to True, the maximum value of the input activation
                will be calculated dynamically, default is True.

        Raises:
            RuntimeError: If the ``dynamic`` and the ``fused_to_weight`` flags are both enabled.
        """
        if dynamic and self.fused_to_weight[0] == 1:
            raise RuntimeError(
                "SmoothQuant cannot be dynamic as scale has been fused to weight already"
            )
        super().set_dynamic(a_dynamic=dynamic, b_dynamic=False)

    @torch.jit.export
    def reset_weight_maxabs(self) -> None:
        """
        Resets weight maxabs.
        """
        self.reset_b_maxabs()

    @property
    def dynamic(self) -> torch.Tensor:
        """
        Checks if the dynamic flag is set for the input activation.

        Returns:
            A boolean tensor set to True if the dynamic flag is one,
                and set to False otherwise.
        """
        return self.a_dynamic

    @property
    def weight_maxabs_computed(self) -> bool:
        """
        Checks if weight_maxabs is already calculated.

        Returns:
            True if weight_maxabs is calculated, False otherwise.
        """
        return self.b_maxabs_exists

    @property
    def input_maxabs_exists(self) -> bool:
        """
        Checks if input_maxabs is already calculated.

        Returns:
            True if input_maxabs is calculated, False otherwise.
        """
        return self.a_maxabs_exists

    def scale_weight(self, wgt):
        """
        Scales weight.

        Args:
            wgt (Tensor): the weight tensor that scaling will be applied on

        Returns:
            scaled weight tensor
        """
        return self.scale_b(wgt).to(wgt.device)

    def scale_input(self, inp):
        """
        Scales the input activation.

        Args:
            inp (Tensor): the input tensor that scaling will be applied on

        Returns:
            scaled input activation tensor
        """
        return self.scale_a(inp).to(inp.device)

    def fuse_to_weight(self, wgt: torch.Tensor) -> None:
        """
        Fuses the scaling factor to the weight tensor.

        Args:
            wgt (Tensor): the weight tensor
        """
        wgt.data = self.scale_weight(wgt.data)
        self.fused_to_weight[0] = 1

    def compute_scale(self, inp_maxabs: torch.Tensor) -> None:
        """
        Computes the scaling tensor.

        Args:
            inp_maxabs (Tensor): the maximum value of absolute of input activation
        """
        super().compute_scale(inp_maxabs, self.weight_maxabs)

    def forward(self, inp: torch.Tensor, wgt: torch.Tensor) -> None:
        """
        Computes the smoothQuant scaling tensor and scales input activation and weight

        Args:
            inp (tensor): the input activation tensor
            wgt (tensor): the weight tensor
        """
        with torch.no_grad():
            if not self.weight_maxabs_computed:
                self.weight_maxabs = self._maxabs(wgt, self.win_ch_axis)
            if not self.input_maxabs_exists or self.dynamic[0] == 1:
                self.input_maxabs = self._maxabs(inp, self.ch_axis)
            else:
                self.input_maxabs = torch.maximum(
                    self._maxabs(inp, self.ch_axis), self.input_maxabs
                )
            self.compute_scale(self.input_maxabs)

    def extra_repr(self) -> str:
        """
        Returns the extra representation of Activation x Weight smoothQuant
        """
        return f"migration_strength = {self.migration_strength.item()}, ch_axis = {self.ch_axis}, win_ch_axis = {self.win_ch_axis}, scale_format = {self.scale_cast.format}, dynamic = {self.dynamic.bool().item()}"
