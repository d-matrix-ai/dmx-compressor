from typing import Optional
import transformers
from transformers import pipeline as hfpipeline
from .model import Model, DmxConfig
import evaluate
from datasets import load_dataset
from huggingface_hub import hf_hub_download

TASK_TO_INPUT_NAMES_LUT = {
    "text-generation": [
        "input_ids",
        "labels",
    ]
}
HF_MODEL_ATTRIBUTE_TO_RETAIN = [
    "name_or_path",
    "config",
    "generation_config",
    "hf_device_map",
    "can_generate",
]


def get_config_file(repo_name, revision, config_name):
    filename = f"configs/{config_name}.yaml"
    try:
        return hf_hub_download(repo_id=repo_name, filename=filename, revision=revision)
    except:
        return None


def dmx_transform(pipe, dmx_config_name):
    config_file = get_config_file(pipe.model_name, pipe.revision, dmx_config_name)
    if config_file is not None:
        config = DmxConfig.from_yaml(config_file)
        pipe.model.transform(config["model"])
    else:
        if dmx_config_name in ["BASELINE", "BASIC"]:
            from . import config_rules

            pipe.model.transform(
                pipe.baseline_config, *eval(f"config_rules.{dmx_config_name}")
            )
        else:
            raise RuntimeError(f"illegal dmx_config: {dmx_config_name}")


def eval_text_generation(
    model,
    dataset,
    metric,
    revision,
    column_name=None,
    dataset_version=None,
    dataset_split="test",
):
    dataset_column_mapping = {
        "wikitext": "text",
        "ptb_text_only": "sentence",
        "lambada": "text",
        "EleutherAI/lambada_openai": "text",
        # Add more datasets and their respective column names here
    }

    if not column_name and dataset in dataset_column_mapping:
        column_name = dataset_column_mapping[dataset]
    if not column_name:
        raise ValueError(
            f"Column name not found for dataset '{dataset}'. Please provide the column_name."
        )

    metric = evaluate.load(metric, module_type="metric")
    dataset = load_dataset(dataset, dataset_version, split=dataset_split)
    results = metric.compute(
        model=model, revision=revision, references=dataset[column_name]
    )
    return results


def pipe_eval(
    model,
    dataset,
    metric,
    revision,
    task,
    column_name=None,
    dataset_version=None,
    dataset_split="test",
):
    task_eval_mapping = {
        "text-generation": eval_text_generation,
        # Add more tasks here
    }

    if task not in task_eval_mapping:
        raise ValueError(f"Unsupported task type '{task}'.")

    eval_function = task_eval_mapping[task]
    return eval_function(
        model, dataset, metric, revision, column_name, dataset_version, dataset_split
    )


def pipeline(
    *args,
    dmx_config: Optional[str] = None,
    trust_remote_code: bool = True,
    device_map: Optional[str] = "auto",
    **kwargs,
):
    kwargs.update(
        {
            "trust_remote_code": trust_remote_code,
            "device_map": device_map,
            # "input_names": TASK_TO_INPUT_NAMES_LUT[kwargs["task"]],
        }
    )
    pipe = hfpipeline(*args, **kwargs)
    pipe.task = kwargs.get("task")
    pipe.model_name = kwargs.get("model")
    pipe.revision = kwargs.get("revision", "main")
    pipe.model = Model(
        pipe.model,
        hf=True,
        input_names=TASK_TO_INPUT_NAMES_LUT[pipe.task],
        attributes_to_retain=HF_MODEL_ATTRIBUTE_TO_RETAIN,
    )
    pipe.baseline_config = pipe.model.dmx_config
    pipe.evaluate = lambda metric, dataset, column_name=None, dataset_version=None, dataset_split="test": pipe_eval(
        pipe.model.body,
        dataset,
        metric,
        pipe.revision,
        pipe.task,
        column_name,
        dataset_version,
        dataset_split,
    )

    dmx_transform(pipe, dmx_config)

    return pipe
