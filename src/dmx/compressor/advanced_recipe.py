from abc import ABC
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from typing import Callable, Optional, List
import torch
from skopt.space import Space
from dmx.compressor.modeling.nn import DmxModule
from dmx.compressor.numerical.observer import (
    HistogramObserver,
    ObserverBase,
)


class DmxBaseRecipe(ABC):
    r"""
    This is an abstract class of ADVANCED mode recipe.
    """

    def __init__(self, hp_gen: Callable, **kwargs):
        super().__init__()
        self.generate_hyperparams = hp_gen
        self.recipe_context_manager = None

    @contextmanager
    def applied_to(self, _model, save_checkpoint_to: Optional[str] = None):
        _hyperparams = self.generate_hyperparams(_model)
        with ExitStack() as stack:
            try:
                yield [
                    stack.enter_context(self.recipe_context_manager(_m, _p))
                    for _m, _p in _hyperparams.items()
                ]
            finally:
                if hasattr(
                    _model, "_save_specific_layers_state_dict_and_register_urls"
                ):
                    _model._save_specific_layers_state_dict_and_register_urls(
                        _hyperparams.keys(), save_checkpoint_to
                    )


@dataclass
class DmxQuantizerCalibrationHyperparams:
    r"""
    Fake quantizer (i.e. CastTo) calibration hyperparameters with default values
    """

    observer_cls: ObserverBase = HistogramObserver
    qscheme_to_overload: torch.qscheme = torch.per_tensor_symmetric
    group_size: Optional[int] = None
    ch_axis: Optional[int] = None


@dataclass
class DmxModuleQuantizerCalibrationHyperparams:
    r"""
    DmxModule boundary cast quantizers calibration hyperparameters with default values
    """

    inputs: Optional[List[DmxQuantizerCalibrationHyperparams]] = None
    outputs: Optional[List[DmxQuantizerCalibrationHyperparams]] = None
    weight: Optional[DmxQuantizerCalibrationHyperparams] = None
    weight_storage: Optional[DmxQuantizerCalibrationHyperparams] = None


@dataclass
class DmxModuleSmoothQuantHyperparams:
    r"""
    DmxModule SmoothQuant hyperparameters with default values
    """

    migration_strength: float = 0.5
    fuse_to_weight: bool = False


@dataclass
class DmxModuleGPTQHyperparams:
    r"""
    DmxModule GPTQ hyperparameters with default values
    """

    microblock_size: int = 1
    block_size: int = 128
    percdamp: float = 0.01


@dataclass
class DmxApproximationFunctionTuningHyperparams:
    r"""
    Approximation function extra_params tuning hyperparameters with default values
    """

    search_space: Optional[Space] = None


class DmxQuantizerCalibrationRecipe(DmxBaseRecipe):
    r"""
    Fake quantizer calibration recipe
    """

    def __init__(self, hp_gen, **kwargs):
        super().__init__(hp_gen, **kwargs)
        self.recipe_context_manager = DmxModule.calibrating_quantizers


class DmxSmoothQuantRecipe(DmxBaseRecipe):
    r"""
    SmoothQuant recipe
    """

    def __init__(self, hp_gen, **kwargs):
        super().__init__(hp_gen, **kwargs)
        self.recipe_context_manager = DmxModule.calibrating_smoothquant


class DmxGPTQRecipe(DmxBaseRecipe):
    r"""
    GPTQ recipe
    """

    def __init__(self, hp_gen, **kwargs):
        super().__init__(hp_gen, **kwargs)
        self.recipe_context_manager = DmxModule.optimal_brain_compressing


class DmxApproximationFunctionTuningRecipe(DmxBaseRecipe):
    r"""
    Approximation function extra_params tuning recipe
    """

    def __init__(self, hp_gen, **kwargs):
        super().__init__(hp_gen, **kwargs)
        self.recipe_context_manager = DmxModule.tuning_approximation_function
