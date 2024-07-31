import os
import torch
from dmx.compressor import dmx
from dmx.compressor.dmx import pipeline, DmxModel
from transformers.modeling_utils import get_parameter_device


pipe = pipeline(
    task="text-generation",
    model="d-matrix/Llama3",
    revision="llama3-8b",
    dmx_config="BASELINE",
    # use_auth_token=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    trust_remote_code=True,
)
seq_len = 1024
x = torch.randint(1, 100, (1, seq_len)).to("cuda")
pipe.model(x)
torch.cuda.empty_cache()

DmxModel.create_submod_transform_forward(pipe.model, "model.layers.15")
DmxModel.create_submod_transform_forward(pipe.model, "model.layers.15.mlp")
DmxModel.create_submod_transform_forward(pipe.model, "model.layers.15.self_attn")

device = get_parameter_device(pipe.model.model.layers[15])

transformer_output0 = pipe.model.model(x)

# use output of transformer as test input for the submodules
submod_input = transformer_output0[0]

# test transformer block
causal_mask = torch.full(
    (1, 1, seq_len, pipe.model.config.max_position_embeddings),
    fill_value=-torch.inf,
    device=device,
)
causal_mask = torch.triu(causal_mask, diagonal=1)
position_ids = torch.range(0, x.shape[1] - 1).unsqueeze(0).to(device)
output0 = pipe.model.model.layers[15](
    submod_input,
    position_ids=position_ids,
    attention_mask=causal_mask,
)
output = pipe.model.model.layers[15].transformed_forward(
    submod_input,
    position_ids=position_ids,
    attention_mask=causal_mask,
)
assert torch.all(output0[0] == output[0])

# test transformer mlp
output0 = pipe.model.model.layers[15].mlp(submod_input)
output = pipe.model.model.layers[15].mlp.transformed_forward(submod_input)
assert torch.all(output0[0] == output[0])

# test transformer attn
output0 = pipe.model.model.layers[15].self_attn(
    submod_input,
    attention_mask=causal_mask,
    position_ids=position_ids,
)
output = pipe.model.model.layers[15].self_attn.transformed_forward(
    submod_input,
    attention_mask=causal_mask,
    position_ids=position_ids,
)
assert torch.all(output0[0] == output[0])
print("\ntests passed: unquantized submods are same as original submods!\n")


bfp16 = "BFP[8|8]{64,-1}(SN)"
rules = (
    dmx.DmxConfigRule(
        module_types=(dmx.nn.Linear,),
        module_config=dict(
            input_format=bfp16,
            weight_format=bfp16,
            bias_format=bfp16,
            output_format=bfp16,
        ),
    ),
)
pipe.model.configure(None, *rules)


# test transformer block
output0 = pipe.model.model.layers[15](
    submod_input,
    position_ids=position_ids,
    attention_mask=causal_mask,
)
output = pipe.model.model.layers[15].transformed_forward(
    submod_input,
    position_ids=position_ids,
    attention_mask=causal_mask,
)
assert torch.any(output0[0] != output[0])

# test transformer mlp
output0 = pipe.model.model.layers[15].mlp(submod_input)
output = pipe.model.model.layers[15].mlp.transformed_forward(submod_input)
assert torch.any(output0[0] != output[0])

# test transformer attn
output0 = pipe.model.model.layers[15].self_attn(
    submod_input,
    attention_mask=causal_mask,
    position_ids=position_ids,
)
output = pipe.model.model.layers[15].self_attn.transformed_forward(
    submod_input,
    attention_mask=causal_mask,
    position_ids=position_ids,
)
assert torch.any(output0[0] != output[0])
print("\ntests passed: quantized submods are different from original submods!\n")
