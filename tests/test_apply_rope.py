import pytest
import torch
import transformers
import dmx
from dmx.compressor.modeling import nn as dmxnn

RANDOM_SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)


@pytest.mark.parametrize("bsz", [1, 8])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("seq_len", [64,512])
@pytest.mark.parametrize("nheads", [4, 8,32])
@pytest.mark.parametrize("unsqueeze_dim", [1,2])
def test_apply_rope(bsz, dim,seq_len,nheads,unsqueeze_dim):
    assert unsqueeze_dim in [1,2]
    if unsqueeze_dim == 1:
        q = torch.rand(bsz,nheads,seq_len,dim).to(device)
        k = torch.rand(bsz,nheads,seq_len,dim).to(device)        
    elif unsqueeze_dim == 2:
        q = torch.rand(bsz,seq_len,nheads,dim).to(device)
        k = torch.rand(bsz,seq_len,nheads,dim).to(device)        

    cos = torch.rand(bsz,seq_len,dim // 2).repeat(1,1,2).to(device)
    sin = torch.rand(bsz,seq_len,dim // 2).repeat(1,1,2).to(device)

    
    gt_q_pos,gt_k_pos = transformers.models.llama.modeling_llama.apply_rotary_pos_emb(
        q,k,cos,sin,unsqueeze_dim = unsqueeze_dim)

    dmx_rope = dmxnn.ApplyRotaryPosEmb()
    dmx_q_pos,dmx_k_pos = dmx_rope(q,k,cos,sin,unsqueeze_dim = unsqueeze_dim)

    rope_config = [x for x in dmx.compressor.config_rules.BASIC if \
                   type(dmx_rope) in x.module_types][0].module_config
    dmx_rope.configure(rope_config)

    basic_q_pos,basic_k_pos = dmx_rope(q,k,cos,sin,unsqueeze_dim = unsqueeze_dim)
    

    atol = 1e-12


    assert torch.allclose(gt_q_pos, dmx_q_pos, atol=atol)
    assert torch.allclose(gt_k_pos, dmx_k_pos, atol=atol)    

    atol = 1e-2    
    assert torch.allclose(gt_q_pos, basic_q_pos, atol=atol)
    assert torch.allclose(gt_k_pos, basic_k_pos, atol=atol)    
    

