from dmx.compressor.modeling.hf import pipeline
from dmx.compressor.modeling import DmxConfigRule, nn
import torch

pipe = pipeline(
    task="text-generation",
    model="d-matrix/opt",
    revision="opt-125m",
    dmx_config="BASELINE",
    trust_remote_code=True,
    device_map="auto",  # enabling model parallel on multi-GPU nodes
)
bfp16 = "BFP[8|8]{64}(SN)"
bfp14 = "BFP[6|8]{64}(SN)"
rules = (
    DmxConfigRule(
        module_types=(nn.Embedding,),
        module_config=dict(
            input_formats=[bfp16],
            output_format=bfp16,
        ),
    ),
)
pipe.model.configure(None, *rules)

x = torch.ones((1, 1024), dtype=int).to("cuda")

model_inputs = {
    "input_ids": torch.tensor(
        [[2, 11475, 2115, 10, 86, 11, 10, 1212, 444, 6, 444, 409]], device="cuda:0"
    ),
    "labels": torch.tensor(
        [[2, 11475, 2115, 10, 86, 11, 10, 1212, 444, 6, 444, 409]], device="cuda:0"
    ),
    "past_key_values": None,
    "use_cache": True,
}
y = pipe.model(**model_inputs)
y = pipe.model(x, labels=x)
breakpoint()
model_inputs["past_key_values"] = [
    (
        torch.empty(
            (
                1,
                pipe.model.model.decoder.layers[i].self_attn.num_heads,
                model_inputs["input_ids"].shape[1],
                pipe.model.model.decoder.layers[i].self_attn.head_dim,
            )
        ).to(pipe.model.model.decoder.layers[i].self_attn.k_proj.weight.device),
        torch.empty(
            (
                1,
                pipe.model.model.decoder.layers[i].self_attn.num_heads,
                model_inputs["input_ids"].shape[1],
                pipe.model.model.decoder.layers[i].self_attn.head_dim,
            )
        ).to(pipe.model.model.decoder.layers[i].self_attn.v_proj.weight.device),
    )
    for i in range(len(pipe.model.model.decoder.layers))
]
y = pipe.model(**model_inputs)

# unquantize the model
rules = (
    DmxConfigRule(
        module_types=(nn.Embedding,),
        module_config=dict(
            input_formats=["SAME"],
            output_format="SAME",
        ),
    ),
)
pipe.model.configure(None, *rules)

# generation
prompt = "Once upon a time in a land far, far away"
generated_texts = pipe(prompt, max_length=50, num_return_sequences=1)
print(generated_texts)

# evaluation
metric = pipe.evaluate(
    "d-matrix/dmx_perplexity",
    dataset="wikitext",
    dataset_version="wikitext-2-raw-v1",
)
print(metric)
