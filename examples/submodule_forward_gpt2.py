import torch
from dmx.compressor.modeling.hf import pipeline, DmxModel
from dmx.compressor.modeling import DmxConfigRule, nn


pipe = pipeline(
    task="text-generation",
    model="d-matrix/gpt2",
    revision="distilgpt2",  # distilgpt2 with gelu instead of gelu_new as activation function
    dmx_config="BASELINE",
    device_map="cuda:0",
    # use_auth_token=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    trust_remote_code=True,
)
x = torch.randint(1, 100, (1, 1024)).to("cuda")
y = pipe.model(x)

DmxModel.create_submod_transform_forward(pipe.model, "transformer")
DmxModel.create_submod_transform_forward(pipe.model, "transformer.h.0")
DmxModel.create_submod_transform_forward(pipe.model, "transformer.h.0.mlp")
DmxModel.create_submod_transform_forward(pipe.model, "transformer.h.0.attn")

transformer_output0 = pipe.model.transformer(x)
transformer_output = pipe.model.transformer.transformed_forward(x)
assert torch.all(transformer_output[0] == transformer_output0[0])

# use output of transformer as test input for the submodules
submod_input = transformer_output0[0]

# test transformer block
output0 = pipe.model.transformer.h[0](submod_input)
output = pipe.model.transformer.h[0].transformed_forward(submod_input)
assert torch.all(output0[0] == output[0])

# test transformer mlp
output0 = pipe.model.transformer.h[0].mlp(submod_input)
output = pipe.model.transformer.h[0].mlp.transformed_forward(submod_input)
assert torch.all(output0[0] == output[0])

# test transformer attn
output0 = pipe.model.transformer.h[0].attn(submod_input)
output = pipe.model.transformer.h[0].attn.transformed_forward(submod_input)
assert torch.all(output0[0] == output[0])
print("\ntests passed: unquantized submods are same as original submods!\n")


bfp16 = "BFP[8|8]{64}(SN)"
rules = (
    DmxConfigRule(
        module_types=(nn.Linear,),
        module_config=dict(
            input_format=bfp16,
            weight_format=bfp16,
            bias_format=bfp16,
            output_format=bfp16,
        ),
    ),
)
pipe.model.configure(None, *rules)

transformer_output0 = pipe.model.transformer(x)
transformer_output = pipe.model.transformer.transformed_forward(x)
assert torch.any(transformer_output[0] != transformer_output0[0])

# use output of transformer as test input for the submodules
submod_input = transformer_output0[0]

# test transformer block
output0 = pipe.model.transformer.h[0](submod_input)
output = pipe.model.transformer.h[0].transformed_forward(submod_input)
assert torch.any(output0[0] != output[0])

# test transformer mlp
output0 = pipe.model.transformer.h[0].mlp(submod_input)
output = pipe.model.transformer.h[0].mlp.transformed_forward(submod_input)
assert torch.any(output0[0] != output[0])

# test transformer attn
output0 = pipe.model.transformer.h[0].attn(submod_input)
output = pipe.model.transformer.h[0].attn.transformed_forward(submod_input)
assert torch.any(output0[0] != output[0])
print("\ntests passed: quantized submods are different from original submods!\n")
