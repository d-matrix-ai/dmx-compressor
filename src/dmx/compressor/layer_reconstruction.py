import math
from typing import Optional, Dict, Any
import torch
import warnings
from contextlib import contextmanager
from dmx.compressor.numerical.observer import HistogramObserver


class LayerReconstructionMixin:
    r"""
    This mixin equips DmxModule with layer-reconstruction functionalities.
    Layer-reconstruction is any post-training process by which
    certain module parameters are fitted to optimize a local objective,
    usually by passing data (input activations) through the module.
    Examples are traditional static activation calibration,
    static SmoothQuant calibration, Optimal Brain Compression, etc.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.obc = None

    def _has_weight(self) -> bool:
        return "weight" in [n for n, _ in self.named_parameters()]

    def set_activation_calibrator(
        self,
        hyperparams: Dict[str, Dict[str, Any]] = None,
    ):
        """
        This function sets the activation calibrators.
        Args:
            hyperparams: a dictionary mapping input_casts keys to a dictionary of calibration parameters.
            Dictionary of calibration parameters can contain keys "observer_cls", "qscheme_to_overload","group_size","ch_axis";
            if unspecified, their value defaults to HistogramObserver, None, None, None
            if hyperparams is None, default is to set input_casts.input_cast to HistogramObserver.
            Example value for hyperparams:
            hyperparams={
                "input_cast": dict(
                    observer_cls=HistogramObserver,
                    qscheme_to_overload: torch.per_tensor_symmetric,
                    group_size: None,
                    ch_axis: -1,
                )
                "multiplier_cast": dict(
                    observer_cls=HistogramObserver,
                    qscheme_to_overload: torch.per_channel_affine,
                    group_size: 16,
                    ch_axis: -2,
                )
            }
        """
        if self.input_casts is not None:
            if hyperparams is None and hasattr(self.input_casts, "input_cast"):
                hyperparams = {"input_cast": {}}
                warnings.warn(
                    "No calibration hyperparams specified, defaults to setting HistogramObserver for input_casts.input_cast only."
                )
            if len(hyperparams) != len(self.input_casts):
                warnings.warn(
                    "Number of keys for hyperparams does not match with length of input_casts, calibrators for some CastTos might not be set."
                )
            for k, params in hyperparams.items():
                if hasattr(self.input_casts, k):
                    observer_cls = (
                        HistogramObserver
                        if (
                            "observer_cls" not in params
                            or params["observer_cls"] is None
                        )
                        else params["observer_cls"]
                    )
                    qscheme_to_overload = (
                        params["qscheme_to_overload"]
                        if "qscheme_to_overload" in params
                        else None
                    )
                    group_size = (
                        params["group_size"] if "group_size" in params else None
                    )
                    ch_axis = params["ch_axis"] if "ch_axis" in params else None

                    if ch_axis is not None:
                        self.input_casts[k].ch_axis = self.input_casts[
                            k
                        ].activation_post_process.ch_axis = ch_axis
                    if qscheme_to_overload is not None:
                        self.input_casts[k].qscheme = qscheme_to_overload
                        self.input_casts[k].is_per_channel = (
                            torch.ao.quantization.utils.is_per_channel(
                                qscheme_to_overload
                            )
                        )
                    if group_size is not None:
                        self.input_casts[k].group_size = group_size
                    self.input_casts[k].activation_post_process = observer_cls(
                        dtype=self.input_casts[k].format,
                        qscheme=self.input_casts[k].qscheme,
                        ch_axis=self.input_casts[k].ch_axis,
                    )
                else:
                    raise ValueError(f"{k} is not a key in input_casts!")

        else:
            warnings.warn(
                "cannot set up activation calibration because of a lack of input_casts",
                RuntimeWarning,
            )

    def set_weight_calibrator(
        self,
        observer_cls=HistogramObserver,
        qscheme_to_overload: Optional[torch.qscheme] = None,
        group_size: int = None,
        ch_axis: int = None,
    ):
        if self.weight_cast is not None:
            if ch_axis is not None:
                self.weight_cast.ch_axis = (
                    self.weight_cast.activation_post_process.ch_axis
                ) = ch_axis
            if qscheme_to_overload is not None:
                self.weight_cast.qscheme = qscheme_to_overload
                self.weight_cast.is_per_channel = (
                    torch.ao.quantization.utils.is_per_channel(qscheme_to_overload)
                )
            self.weight_cast.group_size = group_size if group_size else None
            if self.weight_cast.group_size:
                assert torch.ao.quantization.utils.is_per_tensor(
                    qscheme_to_overload
                ), "group quantization is to be used with per tensor quantization"
                group_num = math.ceil(
                    self.weight.shape[self.weight_cast.ch_axis] * 1.0 / group_size
                )
                self.weight_cast.activation_post_processes = [
                    observer_cls(
                        dtype=self.weight_cast.format,
                        qscheme=self.weight_cast.qscheme,
                        ch_axis=self.weight_cast.ch_axis,
                    ).to(self.weight.device)
                    for i in range(group_num)
                ]
            self.weight_cast.activation_post_process = observer_cls(
                dtype=self.weight_cast.format,
                qscheme=self.weight_cast.qscheme,
                ch_axis=self.weight_cast.ch_axis,
            ).to(self.weight.device)
        else:
            warnings.warn(
                "cannot set up weight calibration because of a lack of weight_cast",
                RuntimeWarning,
            )

    def enable_activation_calib(self, state: bool = True) -> None:
        if state:
            self.input_casts.disable_fake_quant()
            self.input_casts.enable_observer()
        else:
            self.input_casts.enable_fake_quant()
            self.input_casts.disable_observer()

    def enable_weight_calib(self, state: bool = True) -> None:
        if self._has_weight():
            if state:
                self.weight_cast.disable_fake_quant()
                self.weight_cast.enable_observer()
            else:
                self.weight_cast.enable_fake_quant()
                self.weight_cast.disable_observer()

    def update_smoothquant_scale(self, input):
        if self.smoothquant is not None:
            self.smoothquant(input, self.effective_weight)

    def set_smoothquant_params(
        self,
        migration_strength: float = 0.5,
    ) -> None:
        if self.smoothquant is not None:
            self.smoothquant.set_migration_strength(migration_strength)

    def enable_smoothquant_calib(self, state: bool = True) -> None:
        if self.smoothquant is not None:
            if self.smoothquant.fused_to_weight[0] == 1:
                raise RuntimeError(
                    "SmoothQuant cannot be calibrated because it has been fused to weight already"
                )
            self.smoothquant.set_dynamic(False)  # only static needs calibration
            self.smoothquant.enable(not state)
            self.smoothquant.calibrating = state

    def enable_optimal_brain_compression(
        self, state: bool = True, **hyperparams
    ) -> None:
        if isinstance(
            self,
            (
                torch.nn.Linear,
                torch.nn.Conv2d,
            ),
        ):
            if state:
                self.obc = OptimalBrainCompressor(self)
                # TODO: Weight Sparsifier should be handled.
                self.input_casts.disable_fake_quant()
                self.weight_cast.disable_fake_quant()
            else:
                self.input_casts.enable_fake_quant()
                self.weight_cast.enable_fake_quant()
                self.obc.apply(**hyperparams)
                self.obc = None

    @contextmanager
    def calibrating_weight(self) -> None:
        self.enable_weight_calib(True)
        yield self
        self.enable_weight_calib(False)

    @contextmanager
    def calibrating_activation(self) -> None:
        self.enable_activation_calib(True)
        yield self
        self.enable_activation_calib(False)

    @contextmanager
    def calibrating_smoothquant(self) -> None:
        self.enable_smoothquant_calib(True)
        yield self
        self.enable_smoothquant_calib(False)

    @contextmanager
    def optimal_brain_compressing(self, **hyperparams) -> None:
        self.enable_optimal_brain_compression(True, **hyperparams)
        yield self
        self.enable_optimal_brain_compression(False, **hyperparams)


class OptimalBrainCompressor:
    H: Optional[torch.Tensor] = None

    def __init__(self, module):
        self.module = module
        self.example_counter = 0

    def measure_hessian(self, inp):
        # TODO: clean up ugliness
        if self.H is None:
            self.H = torch.zeros(1).to(inp.device)
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.module, torch.nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.module, torch.nn.Conv2d):
            inp = torch.nn.functional.unfold(
                inp,
                self.module.kernel_size,
                dilation=self.module.dilation,
                padding=self.module.padding,
                stride=self.module.stride,
            )
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.example_counter / (self.example_counter + tmp)
        self.example_counter += tmp
        inp = math.sqrt(2 / self.example_counter) * inp.float()
        self.H = self.H + inp.matmul(inp.t())

    def apply(self, microblock_size=1, block_size=128, percdamp=0.01):
        assert block_size % microblock_size == 0
        if self.module.weight_sparsifier.sparseness.blocked:
            assert (
                microblock_size % self.module.weight_sparsifier.sparseness.block_size
                == 0
            )
        if self.module.weight_cast.format.blocked:
            assert microblock_size % self.module.weight_cast.format.block_size == 0

        # TODO: clean up ugliness
        W = self.module.weight.data.clone()
        if isinstance(self.module, torch.nn.Conv2d):
            W = W.flatten(1)
        W = W.float()
        ncols = W.shape[1]

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(ncols, device=H.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # TODO: allow OBC candidate selection

        for i1 in range(0, ncols, block_size):
            i2 = min(i1 + block_size, ncols)
            count = i2 - i1

            _W = W[:, i1:i2].clone()
            _Q = torch.zeros_like(_W)
            _E = torch.zeros_like(_W)
            _Hinv = Hinv[i1:i2, i1:i2]

            for j1 in range(0, count, microblock_size):
                j2 = min(j1 + microblock_size, count)
                w = _W[:, j1:j2]
                hinv = _Hinv[j1:j2, j1:j2]
                q = self.module.weight_hypernet(w)
                err = (w - q).matmul(torch.linalg.inv(hinv))
                _Q[:, j1:j2] = q
                _W[:, j2:] -= err.matmul(_Hinv[j1:j2, j2:])
                _E[:, j1:j2] = err

            Q[:, i1:i2] = _Q
            W[:, i2:] -= _E.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        self.module.weight.data = Q.reshape(self.module.weight.shape).to(
            self.module.weight.data.dtype
        )
