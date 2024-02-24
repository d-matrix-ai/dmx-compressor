from typing import Any, Optional
import transformers
from transformers import pipeline as hfpipeline
from .model import Model
import requests
import evaluate
import os
import yaml
from datasets import load_dataset

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

# from huggingface_hub import hf_hub_download

# def download_config_from_hf_hub(model_name, config_name, revision):
#     # This will download the file to a cache and return the file path
#     file_path = hf_hub_download(repo_id=model_name, filename=config_name, revision=revision)
#     return file_path


def load_config(model_name, config_name, revision):
    repo_id = model_name
    token = os.environ.get("HUGGINGFACE_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    config_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/configs/{config_name}.yaml"
    response = requests.get(config_url, headers=headers)
    response.raise_for_status()

    return yaml.safe_load(response.content)


def eval(model, metric, dataset, column_name, dataset_version=None, dataset_split="test"):
    metric = evaluate.load(metric, module_type="metric")
    dataset = load_dataset(dataset, dataset_version, split=dataset_split)
    results = metric.compute(model=model, references=dataset[column_name])
    return results


def pipeline(
    *args,
    dmx_config: Optional[str] = None,
    **kwargs,
):
    pipe = hfpipeline(*args, **kwargs)
    revision = kwargs.get("revision", "main")
    config = (
        load_config(kwargs.get("model"), dmx_config, revision) if dmx_config else None
    )
    # config_file_path = download_config_from_hf_hub("d-matrix/gpt2", "configs/BASIC.yaml", "temp-distilgpt2")
    pipe.model = Model(
        pipe.model, hf=True, input_names=task_input_name_lookup[type(pipe)]
    )
    pipe.eval = (
        lambda metric, dataset, column_name, dataset_version=None, dataset_split="test": eval(
            pipe.model.body, metric, dataset, column_name, dataset_version, dataset_split
        )
    )
    pipe.model.transform(config)
    return pipe
