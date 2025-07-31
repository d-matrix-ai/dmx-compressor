import torch
import copy
from dmx.compressor.modeling import DmxModel
from dmx.compressor.plugins import PluginBase,PluginLayerData,ActivatePlugins
import numpy as np
import skopt
from skopt import gp_minimize
from skopt.space import Categorical
from dmx.compressor import nn
from functools import partial

class NW(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lnorm1 = torch.nn.LayerNorm(10)
        self.lnorm2 = torch.nn.LayerNorm(10)
    def forward(self,x,branch):
        x = self.lnorm1(x)
        if branch:
            x = self.lnorm2(x)
        return x

class SlancPlugin(PluginBase):
    def process_model(self,model):
        #Pretend this is applying SLANC
        model._gm.lnorm1.approximator.function =  copy.deepcopy(model._gm.lnorm1.approximator.function)
        model._gm.lnorm1.approximator.function.extra_params.update({'norm' : 10})

        model._gm.lnorm2.approximator.function =  copy.deepcopy(model._gm.lnorm2.approximator.function)
        model._gm.lnorm2.approximator.function.extra_params.update({'norm' : 11})
        
    def process_layer(self,layer_data):
        #We do not do any calibration
        pass 


class CalibrationPlugin(PluginBase):
    def __init__(self,search_space_dict):
        self.search_space_dict = search_space_dict
    def process_model(self,model):
        pass
    def _optimize_layer(self,mod,input,args,kwargs,search_space):
        mod.approximator.function = copy.deepcopy(
            mod.approximator.function
        )
        solver = partial(gp_minimize, n_calls=20)
        @skopt.utils.use_named_args(search_space)
        def obj_func(**extra_params):
            mod.approximator.function.extra_params.update(extra_params)
            _ = mod(input, *args, **kwargs)
            _e = mod.approximation_error
            return torch.nn.functional.mse_loss(_e, torch.zeros_like(_e)).item()

        _res = solver(obj_func, search_space)
        mod.approximator.function.extra_params.update(
            {_p.name: _opt for _p, _opt in zip(search_space, _res.x)}
        )
        
    def process_layer(self,layer_data):
        for lay_type,search_space in self.search_space_dict.items():
            if isinstance(layer_data.mod,lay_type):
                self._optimize_layer(layer_data.mod,
                                     layer_data.input_before_cast,
                                     layer_data.args,
                                     layer_data.kwargs,
                                     search_space)
                
    
model = NW()
DmxModel.from_torch(model)
model.to_basic_mode()
model(torch.rand(1,10),True)

search_space_dict = {torch.nn.LayerNorm : [Categorical(
    np.logspace(start=0.1, stop=10, num=7, base=2), 
    name="norm",
)]}

with ActivatePlugins(CalibrationPlugin(search_space_dict)).applied_to(model):
    model(torch.rand(1,10),True)
print(model.dmx_config)

with ActivatePlugins(SlancPlugin()).applied_to(model):
    print('SLANC done!')
print(model.dmx_config)
