import pytest
import torch
from dmx.compressor import DmxModel
from dmx.compressor.numerical.cast import CastTo
from dmx.compressor import format,config_rules


RANDOM_SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)


def test_flexible_quant():
    class TestNW(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = torch.nn.Linear(10,20,bias = False)
        def forward(self,x):
            return self.lin1(x) + self.lin1(x)

    model = TestNW()
    model = DmxModel.from_torch(model)
    model.to_baseline_mode()
    test_inp = torch.rand(5,10)
    model(test_inp)        

    bfp16_cast = CastTo(format=format.BFP16A_64, block_dim=-1)
    fp16_cast = CastTo(format=format.FLOAT16, block_dim=-1)
    int8_cast = CastTo(format=format.INT8, block_dim=-1)

    ##The Model builer models sometimes quantize weights to FP16 before quantizing to BFP16_64
    lin1_module_config=dict(
        input_formats=[format.BFP16A_64],
        weight_format=format.BFP16A_64,
        output_formats=[format.FLOAT16],
        pre_weight_transform = {'format' : format.FLOAT16}
    )
    model._gm.lin1.configure(lin1_module_config)
    lin1_res = model._gm.lin1(test_inp)
    gt_lin1_res = fp16_cast(bfp16_cast(test_inp) @ bfp16_cast(fp16_cast(model.lin1.weight)).T)
    assert torch.all(lin1_res == gt_lin1_res)




    ##The Model builer models sometimes reshape the weight matrix (for conv2d for example before quantizing),
    ##Reshaping affects the result of block quantization methods such as BFP16_64
    lin1_module_config=dict(
        input_formats=[format.BFP16A_64],
        weight_format=format.BFP16A_64,
        output_formats=[format.FLOAT16],
        pre_weight_transform = {'format' : format.FLOAT16,
                                'shaping': [('permute',(1,0)),
                                            ('view',(10,5,4))
                                            ]
                                }

    )
    model._gm.lin1.configure(lin1_module_config)
    lin1_res = model._gm.lin1(test_inp)
    weight_reshaped = model.lin1.weight.permute(1,0).view(10,5,4)
    weight_quantized = bfp16_cast(fp16_cast(weight_reshaped))
    weight_orig_size = weight_quantized.view(10,20).permute(1,0)

    gt_lin1_res = fp16_cast(bfp16_cast(test_inp) @ weight_orig_size.T)
    assert torch.all(lin1_res == gt_lin1_res)



    ##The Model builer models sometimes compute part of the tensor (that does not depend on the input)
    ##on the host in FP32. We need to skip quantizationm 
    resadd_module_config=dict(
        input_formats=[format.INT8,format.INT8],
        output_formats=[format.INT8],
        pre_input_transform = [{'noquant_shortcut' : [slice(0,1)]},
                               {'noquant_shortcut' : [slice(0,1)]}],
        pre_output_transform = [{'noquant_shortcut' : [slice(0,1)]}]

    )
    model._gm.resadd.configure(resadd_module_config)
    lhs = rhs =  model._gm.lin1(test_inp)
    resadd_result = model._gm.resadd(lhs,rhs)
    gt_resadd_result = int8_cast(int8_cast(lhs)+int8_cast(rhs))
    gt_resadd_result[0] = lhs[0] + rhs[0]
    assert torch.all(resadd_result == gt_resadd_result)
