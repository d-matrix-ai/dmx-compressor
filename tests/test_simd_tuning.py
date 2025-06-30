import pytest
import torch
import torch.nn as nn
import numpy as np
from dmx.compressor import DmxModel
from dmx.compressor.modeling import nn as dmxnn
from skopt.space import Categorical

from dmx.compressor.advanced_recipe import (
    DmxApproximationFunctionTuningHyperparams,
    DmxApproximationFunctionTuningRecipe,
)

from dmx.compressor.advanced_recipe import DmxModuleSmoothQuantHyperparams


class TestNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(40,50)
        self.lnorm1 = torch.nn.LayerNorm(50)
        self.lin2 = nn.Linear(50,60)
        self.lnorm2 = torch.nn.LayerNorm(60)
    def forward(self,x):
        x = self.lnorm1(self.lin1(x))
        x = torch.nn.functional.relu(x)
        return self.lnorm2(self.lin2(x))

RANDOM_SEED = 0

torch.manual_seed(RANDOM_SEED)

def test_simd_tuning(
):
    model = TestNetwork()
    model = DmxModel.from_torch(model)
    model.to_basic_mode()
    model(torch.randn(10,40))
    
    def hp_gen(_model):
        return {
            _m: DmxApproximationFunctionTuningHyperparams(
                search_space=[
                    Categorical(
                        np.logspace(start=-5, stop=1, num=7, base=2), 
                        name="norm",
                    )
                ],
            )
            for _, _m in _model.named_dmx_modules()
            if isinstance(_m, (nn.LayerNorm,))
        }
    with DmxApproximationFunctionTuningRecipe(hp_gen).applied_to(model):
        model(torch.randn(10,40))
    
    assert model._gm.lnorm1.approximator.function is not \
        model._gm.lnorm2.approximator.function
    
    assert 'norm' in model._gm.lnorm1.approximator.function.extra_params
    assert 'norm' in model._gm.lnorm2.approximator.function.extra_params
