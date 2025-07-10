import math
import torch
import skopt
from skopt import gp_minimize
import numpy as np
import copy
from typing import Optional
from functools import partial
from contextlib import contextmanager
from dmx.compressor.numerical.observer import HistogramObserver
from dmx.compressor.functional.approximate import NoApproximation


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
        self.aft = None
        self.obc = None

    def _has_weight(self) -> bool:
        return "weight" in [n for n, _ in self.named_parameters()]

    def update_smoothquant_scale(self, input):
        if self.smoothquant is not None:
            self.smoothquant(input, self.effective_weight)

    def enable_quantizer_calib(self, state: bool, hyperparams) -> None:
        if hyperparams.inputs is not None:
            for _k in self.input_casts.keys():
                self.input_casts[_k].enable_calibration(
                    state, **hyperparams.inputs[_k].__dict__
                )
        if hyperparams.outputs is not None:
            for _k in self.output_casts.keys():
                self.output_casts[_k].enable_calibration(
                    state, **hyperparams.outputs[_k].__dict__
                )
        if self._has_weight():
            if hyperparams.weight is not None:
                self.weight_cast.enable_calibration(
                    state, **hyperparams.weight.__dict__
                )
            if hyperparams.weight_storage is not None:
                self.weight_storage_cast.enable_calibration(
                    state, **hyperparams.weight_storage.__dict__
                )

    def enable_smoothquant_calib(self, state: bool, hyperparams) -> None:
        if self.smoothquant is not None:
            if self.smoothquant.fused_to_weight[0] == 1:
                raise RuntimeError(
                    "SmoothQuant cannot be calibrated because it has been fused to weight already"
                )
            self.smoothquant.set_migration_strength(hyperparams.migration_strength)
            self.smoothquant.set_dynamic(False)  # only static needs calibration
            self.smoothquant.enable(not state)
            self.smoothquant.calibrating = state
            if not state and hyperparams.fuse_to_weight:
                self.smoothquant.fuse_to_weight(self.weight)

    def enable_optimal_brain_compression(self, state: bool, hyperparams) -> None:
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
                self.obc.apply(**hyperparams.__dict__)
                self.obc = None

    def enable_approximation_function_tuning(self, state: bool, hyperparams) -> None:
        if not isinstance(self.approximation_function, NoApproximation):
            if state:
                self.aft = ApproximationFunctionTuner(self, hyperparams.search_space)
            else:
                self.aft = None

    @contextmanager
    def calibrating_quantizers(self, hyperparams) -> None:
        self.enable_quantizer_calib(True, hyperparams)
        yield self
        self.enable_quantizer_calib(False, hyperparams)

    @contextmanager
    def calibrating_smoothquant(self, hyperparams) -> None:
        self.enable_smoothquant_calib(True, hyperparams)
        yield self
        self.enable_smoothquant_calib(False, hyperparams)

    @contextmanager
    def optimal_brain_compressing(self, hyperparams) -> None:
        self.enable_optimal_brain_compression(True, hyperparams)
        yield self
        self.enable_optimal_brain_compression(False, hyperparams)

    @contextmanager
    def tuning_approximation_function(self, hyperparams) -> None:
        self.enable_approximation_function_tuning(True, hyperparams)
        yield self
        self.enable_approximation_function_tuning(False, hyperparams)

    @contextmanager
    def slanc_tuning(self, hyperparams) -> None:
        if (
            isinstance(self, (torch.nn.LayerNorm, torch.nn.RMSNorm))
            and not isinstance(self.approximation_function, NoApproximation)
            and self.approximation_function.algorithm == "vsimd"
        ):
            prev_ln_weight = (
                hyperparams.prev_ln_weight.detach().cpu().numpy().astype(np.float32)
            )

            if hyperparams.position == "post_attn":
                W_V = (
                    hyperparams.prev_layer.v_proj.get_parameter("weight")
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                P = (
                    hyperparams.prev_layer.out_proj.get_parameter("weight")
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                norm = P @ W_V
                assert norm.shape[0] == norm.shape[1]
                norm += np.eye(norm.shape[0])
                norm *= prev_ln_weight
                # norm *= prev_ln_weight # gives less scale
                norm = np.linalg.norm(norm, ord="fro")
            elif hyperparams.position == "post_mlp":
                A = (
                    hyperparams.prev_layer.fc1.get_parameter("weight")
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                B = (
                    hyperparams.prev_layer.fc2.get_parameter("weight")
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                norm = (
                    np.linalg.norm(prev_ln_weight, ord=1)
                    * np.linalg.norm(A, ord=2)
                    * np.linalg.norm(B, ord=2)
                    / prev_ln_weight.shape[0]
                )
        # Since the approximator is shared between all instances of same module
        # deepcopy to allow each simd module to have different simd parameters
        self.approximator.function = copy.deepcopy(self.approximator.function)

        # SLANC assumes the layernorm input will be divided by norm
        # However, the SIMD kernels multiply the input by the norm parameter
        # Hence the 1/x transformation
        self.approximator.function.extra_params.update({"norm": 1.0 / norm})
        yield self


class ApproximationFunctionTuner:
    def __init__(self, module, search_space):
        self.module = module
        self.search_space = search_space
        self.solver = partial(gp_minimize, n_calls=20)

    def optimize(self, input, *args, **kwargs):
        self.module.approximator.function = copy.deepcopy(
            self.module.approximator.function
        )
        module_aft = self.module.aft
        # To avoid infinite recursion in the module's forward pass
        self.module.aft = None

        @skopt.utils.use_named_args(self.search_space)
        def obj_func(**extra_params):
            self.module.approximator.function.extra_params.update(extra_params)
            _ = self.module(input, *args, **kwargs)
            _e = self.module.approximation_error
            return torch.nn.functional.mse_loss(_e, torch.zeros_like(_e)).item()

        _res = self.solver(obj_func, self.search_space)
        self.module.aft = module_aft
        self.module.approximator.function.extra_params.update(
            {_p.name: _opt for _p, _opt in zip(self.search_space, _res.x)}
        )


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
