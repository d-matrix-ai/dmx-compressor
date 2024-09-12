# This is an example of how to do quantization and calibration for a model
from dmx.compressor.modeling.hf import pipeline
from dmx.compressor.modeling import DmxConfigRule, nn
import torch
from dmx.compressor.numerical.observer import HistogramObserver, MinMaxObserver

# instantiating model
pipe = pipeline(
    task="text-generation",
    model="d-matrix/Llama-2",
    revision="Llama2-7b",
    dmx_config="BASELINE",
    trust_remote_code=True,
    device_map="auto",  # enabling model parallel on multi-GPU nodes. splitting layers evenly across devices based on module size.
)

# creating config rules
# xxx_format takes a single value
# input_formats takes a list or a dictionary. When a list is passed, the formats will be set in the order of the castTos within input_casts.
format = "XP[8,0](CSN)"
rules = (
    DmxConfigRule(
        module_types=(nn.Linear,),
        module_config=dict(
            input_formats=[format],  # option 1
            # input_formats = {"input_cast": format} # option 2
            weight_format=format,
        ),
    ),
    DmxConfigRule(
        module_types=(nn.ScaledDotProductAttention,),
        module_config=dict(
            input_formats=[format, format, format],  # option 1
            # input_formats={
            #     "query_states_cast": format,
            #     "key_states_cast": format,
            #     "value_states_cast": format,
            #     "attn_mask_cast": None,
            # },  # option 2
            weight_format=format,
        ),
    ),
    DmxConfigRule(
        module_types=(nn.ActActMatMul,),
        module_config=dict(
            input_formats=[format, format],  # option 1
            # input_formats = {"input_cast": format, "multiplier_cast":format} # option 2
        ),
    ),
)
# configure model based on rules
pipe.model.configure(None, *rules)

"""
Note: if the data format does not require calibration, all steps from here until right before evaluation can be skipped
"""
# a forward pass needs to be done before calibration so that JIT transformation is triggered and dmx modules exists
x = torch.randint(1, 100, (1, 1024))
with torch.no_grad():
    y = pipe.model(x)

# To check the transformed model, call pipe.model._gm

# specifying layers to calibrate
calibration_layers_matmul = {
    n: m for n, m in pipe.model.named_dmx_modules() if isinstance(m, nn.ActActMatMul)
}
calibration_layers_lin = {
    n: m for n, m in pipe.model.named_dmx_modules() if isinstance(m, (nn.Linear,))
}
calibration_layers_attention = {
    n: m
    for n, m in pipe.model.named_dmx_modules()
    if isinstance(m, (nn.ScaledDotProductAttention,))
}

# specifying hyperparameters to use for calibration
matmul_hyperparams = {
    "input_cast": dict(
        observer_cls=HistogramObserver,
        qscheme_to_overload=None,
        group_size=None,
        ch_axis=None,
    ),
    "multiplier_cast": dict(
        observer_cls=MinMaxObserver,
        qscheme_to_overload=torch.per_channel_affine,
        ch_axis=-2,
    ),
}

"""
if hyperparams=None, it defaults to
{
    "input_cast": dict(
        observer_cls=HistogramObserver,
        qscheme_to_overload=None,
        group_size=None,
        ch_axis=None,
    ),
}
"""

lin_hyperparams = None

"""
if values of hyperparams are empty dicts, it defaults to 
dict(
    observer_cls=HistogramObserver,
    qscheme_to_overload=None,
    group_size=None,
    ch_axis=None,
)
"""
attention_hyperparams = {
    "query_states_cast": {},
    "key_states_cast": {},
    "value_states_cast": {},
}


# doing calibration
with torch.no_grad(), pipe.model.calibrating_weights(
    calibration_layers_lin.items()
), pipe.model.calibrating_activations(
    calibration_layers_matmul.items(), matmul_hyperparams
), pipe.model.calibrating_activations(
    calibration_layers_lin.items(), lin_hyperparams
), pipe.model.calibrating_activations(
    calibration_layers_attention.items(), attention_hyperparams
):
    pipe.model(x, labels=x)


# do evaluation
metric = pipe.evaluate(
    "d-matrix/dmx_perplexity",
    dataset="wikitext",
    dataset_version="wikitext-2-raw-v1",
)
print(metric)
