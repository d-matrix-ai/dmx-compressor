from typing import Any, Optional
import transformers
from transformers import pipeline as hfpipeline
from .model import Model

task_input_name_lookup = {
    transformers.pipelines.text_generation.TextGenerationPipeline: [
        "input_ids",
        "labels",
    ]
}

# dmx_config_dicts = SimpleNamespace(
#     **{
#         os.path.splitext(os.path.basename(config_file))[0]: config_file
#         for config_file in glob.glob("./configs/*.yaml")
#     }
# )


def evaluate(metric, dataset):
    metric.compute(model, references=input_texts)


def pipeline(
    *args,
    dmx_config: Optional[str] = None,
    **kwargs,
):
    pipe = hfpipeline(*args, **kwargs)
    pipe.evaluate = evaluate


    pipe.model = Model(
        pipe.model, hf=True, input_names=task_input_name_lookup[type(pipe)]
    )
    pipe.model.transform(eval(f"dmx_config_dicts.{dmx_config}"))
    return pipe
