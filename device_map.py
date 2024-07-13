import torch
import mltools.dmx as dmx
from mltools.dmx import pipeline as dmxpipeline
from transformers import AutoConfig

embedding_layers = {
    "bloom": "transformer.word_embeddings",
    "gpt2": "transformer.wte",
}

def get_number_of_params(module):
    return sum(p.numel() for p in module.parameters())

hf_token = "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"

model = "d-matrix/gpt2"
revision = "gpt2"
dmx_config = "BASELINE"

pipe = dmxpipeline(
    task = "text-generation",
    model = model,
    revision = revision,
    dmx_config = dmx_config,
    device_map="auto",
    trust_remote_code=True,
)

hf_device_map = pipe.model.hf_device_map
params = {}
for k, v in hf_device_map.items():
    params[k] = get_number_of_params(pipe.model.get_submodule(k))

print(params)
print("test equality:", pipe.model.lm_head.weight.data_ptr() == pipe.model.transformer.wte.weight.data_ptr())
# print("test equality:", pipe.model.lm_head.weight.data_ptr() == pipe.model.transformer.word_embeddings.weight.data_ptr())
# reorder params such that "lm_head" is last
params = {k: v for k, v in sorted(params.items(), key=lambda item: item[0] == "lm_head")}

device_map = {}
cur_device = 0
params_on_cur_device = 0

total_params = sum(params.values())
num_devices = torch.cuda.device_count()
print(f"Total params: {total_params}, num_devices: {num_devices}")
average_params = total_params // num_devices
for k, v in params.items():
    if params_on_cur_device != 0 and params_on_cur_device + v > average_params * 1.25:
        cur_device += 1
        params_on_cur_device = 0
    device_map[k] = cur_device
    params_on_cur_device += v

print(device_map)