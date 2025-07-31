from dataclasses import dataclass
from torch import Tensor
from typing import Union,List,TYPE_CHECKING,Dict,Any
from abc import ABC,abstractmethod
from contextlib import contextmanager


import dmx.compressor.modeling


tens_or_list_type = Union[Tensor,List[Tensor]]

@dataclass
class PluginLayerData:
    input_before_cast : tens_or_list_type
    input_after_cast : tens_or_list_type
    output_before_cast : tens_or_list_type
    output_after_cast : tens_or_list_type
    mod : "DmxModule"
    args : List[Any]
    kwargs : Dict[str,Any]

    
class PluginBase(ABC):
    @abstractmethod
    def process_model(self,model : "DmxModel"):
        '''
        Define here the model transformations that do not depend on a calibration input.
        For example, layernorm scale computation using SLANC

        '''
        pass

    @abstractmethod
    def process_layer(self, layer_data : PluginLayerData):
        '''
        Define here the transformations that depend on a calibration input.
        This function will be called for each layer (DmxModule) with the layer inputs and
        outputs. These come from the calibration data you call the model with. You have to call
        the model with this calibration data

        '''
        
        pass
    

class ActivatePlugins:
    def __init__(self,plugins : Union[PluginBase,List[PluginBase]]):
        if isinstance(plugins,PluginBase):
            plugins = [plugins]
        self.plugins = plugins

    @contextmanager
    def applied_to(self,model : "DmxModel"):
        for p in self.plugins:
            p.process_model(model)
        dmx.compressor.modeling.DmxModule.plugins = self.plugins
        yield
        dmx.compressor.modeling.DmxModule.plugins = []
        
    
                 
