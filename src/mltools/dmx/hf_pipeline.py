from typing import Any, Optional
import transformers
from transformers import pipeline as hfpipeline
from .model import Model
import requests
import evaluate
import os
import yaml
from datasets import load_dataset
from huggingface_hub import hf_hub_download

task_input_name_lookup = {
    transformers.pipelines.text_generation.TextGenerationPipeline: [
        "input_ids",
        "labels",
    ]
}


def load_config(repo_name, config_name, revision):
    file_path = hf_hub_download(
        repo_id=repo_name, filename=f"configs/{config_name}.yaml", revision=revision
    )
    return file_path


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


def eval(
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
    **kwargs,
):
    pipe = hfpipeline(*args, **kwargs)
    pipe.revision = kwargs.get("revision", "main")
    pipe.task = kwargs.get("task")
    config = load_config(kwargs.get("model"), dmx_config, pipe.revision)
    pipe.model = Model(
        pipe.model, hf=True, input_names=task_input_name_lookup[type(pipe)]
    )
    pipe.model.transform(config)
    pipe.eval = lambda metric, dataset, column_name=None, dataset_version=None, dataset_split="test": eval(
        pipe.model.body,
        dataset,
        metric,
        pipe.revision,
        pipe.task,
        column_name,
        dataset_version,
        dataset_split,
    )

    return pipe
