import torch,pytest,pickle
import numpy as np
from mltools import corsair
from numerics import TestData


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(
    "from_format,to_format,dimension",
    (
                # passed tests
                ("BFP16_64", "FP16", None),
                ("BFP12_128", "FP16", None),
                ("BFP32_1", "BFP16_64", 'row'),
                ("FP16", "BFP12_128", 'row'),
            
                # xfails: lack of test data, issue 171
                pytest.param("BFP24_64", "FP16", None,marks=pytest.mark.xfail),
                pytest.param("FP16", "BFP24_64", None,marks=pytest.mark.xfail),
                
                # xfails: qtorch bug, issue 169
                pytest.param("BFP32_1", "FP16", None,marks=pytest.mark.xfail),
                pytest.param("FP16", "BFP16_64", 'row',marks=pytest.mark.xfail),
                pytest.param("FP16", "BFP16_64", 'col',marks=pytest.mark.xfail),
                pytest.param("FP16", "BFP12_128", 'col',marks=pytest.mark.xfail),
    ),
)

def test_conversion(from_format,to_format,dimension):
    '''
        dimension specifies which dimension is used for blocking when to_format is BFP
        otherwise it should be set None
    '''
    try:
        data = TestData(from_format, to_format, dimension if dimension == 'col' else None)
    except (UnboundLocalError,ValueError) as e:
        assert False, f'lack of data: {from_format} to {to_format}'
        
    x = data.input.to_fp32()
    y = data.output.to_fp32()
    x,y = torch.Tensor(x).to(DEVICE),torch.Tensor(y).to(DEVICE)
    
    if to_format == 'FP16': 
        shorthand = 'FP[1|5|10](N)'
            
    elif to_format.startswith('BFP'):
        nbits = int(to_format.split('_')[0][3:]) # number of bits
        block_size, mantissa = int(to_format.split('_')[1]), nbits - 8
        dim_arg = 1 if dimension == 'row' else '0' 
        shorthand = f'BFP[{mantissa}|8]' + '{'+f'{block_size},{dim_arg}'+'}' + '(N)'
    
    qtorch_y = corsair.CastTo(shorthand)(x)
    mismatch = ((qtorch_y - y)!=0) # mismatch is a boolean arrary of the same size as that of x and y
    
    
    # detailed information of mismatched values 
    err_msg = 'values that qtorch disagrees with matlab conversion:\n'
    err_msg += f'Original Value in {from_format}:\n'
    err_msg += f'{x[mismatch]}\n' # extract all values from x which cause the mismatch
    
    err_msg += f'Desired values (matlab converter) in {to_format}:\n'
    err_msg += f'{y[mismatch]}\n'
    
    err_msg += f'qtorch converted values in {to_format}:\n'
    err_msg += f'{qtorch_y[mismatch]}\n'
    
    
    assert torch.allclose(qtorch_y,y,atol=0), err_msg
