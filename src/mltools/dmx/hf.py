from typing import Optional, Dict
from transformers import pipeline as hfpipeline
from mltools.fx.transform import substitute_transform
import transformers
from .model import DmxModelMixin, DmxConfig
import evaluate
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from .model import DmxModel, DmxConfig
from tqdm import tqdm
import torch


def get_config_file(repo_name, revision, config_name):
    try:
        return hf_hub_download(
            repo_id=repo_name, filename=f"configs/{config_name}.yaml", revision=revision
        )
    except Exception as e:
        print(f"Failed to download the file: {str(e)}")
        return None


def dmx_transform(pipe, dmx_config_name):
    config_file = get_config_file(pipe.model_name, pipe.revision, dmx_config_name)
    if config_file is not None:
        config = DmxConfig.from_yaml(config_file)
        pipe.model.configure(config)
    else:
        if dmx_config_name in ["BASELINE", "BASIC"]:
            from . import config_rules

            # NOTE: assuming pipe.model is in BASELINE mode
            pipe.model.configure(None, *eval(f"config_rules.{dmx_config_name}"))
        else:
            raise RuntimeError(f"illegal dmx_config: {dmx_config_name}")


def prepare_dataset_and_column(
    dataset, column_name=None, dataset_version=None, dataset_split="test", seed=42
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

    dataset = load_dataset(dataset, dataset_version, split=dataset_split)
    if dataset_split == "train":
        dataset = dataset.shuffle(seed=seed)

    return dataset, column_name


def do_forward_on(
    model,
    tokenizer,
    dataset,
    column_name=None,
    dataset_version=None,
    dataset_split="test",
    num_samples=None,
    seed=42,
):
    dataset, column_name = prepare_dataset_and_column(
        dataset, column_name, dataset_version, dataset_split, seed
    )

    encodings = tokenizer("\n\n".join(dataset[column_name]), return_tensors="pt")

    if hasattr(model.config, "max_position_embeddings"):
        max_seq_len = model.config.max_position_embeddings
    elif hasattr(model.config, "n_positions"):
        max_seq_len = model.config.n_positions
    else:
        max_seq_len = 2048

    stride = max_seq_len

    seq_len = encodings.input_ids.size(1)

    if num_samples:
        seq_len = min((num_samples - 1) * stride + max_seq_len, seq_len)
    else:
        seq_len = (seq_len // stride) * stride

    nlls = []
    prev_end_loc = 0
    model.eval()

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_seq_len, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood.to(model.device))

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    loss = torch.stack(nlls).float().sum() / end_loc
    ppl = torch.exp(loss)

    return dict(
        loss=loss.item(),
        perplexity=ppl.item(),
    )


def eval_text_generation(
    model,
    dataset,
    metric,
    revision,
    column_name=None,
    dataset_version=None,
    dataset_split="test",
):
    dataset, column_name = prepare_dataset_and_column(
        dataset, column_name, dataset_version, dataset_split
    )

    metric = evaluate.load(metric, module_type="metric")

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
        }
    )
    pipe = hfpipeline(*args, **kwargs)
    pipe.task = kwargs.get("task")
    pipe.model_name = kwargs.get("model")
    pipe.revision = kwargs.get("revision", "main")
    pipe.model = DmxModel.from_torch(
        pipe.model,
    )
    pipe.evaluate = lambda metric, dataset, column_name=None, dataset_version=None, dataset_split="test": pipe_eval(
        pipe.model,
        dataset,
        metric,
        pipe.revision,
        pipe.task,
        column_name,
        dataset_version,
        dataset_split,
    )

    pipe.do_forward_on = lambda dataset, column_name=None, dataset_version=None, dataset_split="test", num_samples=None, seed=42: do_forward_on(
        pipe.model,
        pipe.tokenizer,
        dataset,
        column_name,
        dataset_version,
        dataset_split,
        num_samples,
        seed,
    )

    dmx_transform(pipe, dmx_config)

    return pipe


class DmxPreTrainedModel(transformers.modeling_utils.PreTrainedModel, DmxModelMixin):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        _model = super().from_pretrained(*args, **kwargs)
        _model.base_model = substitute_transform(_model.base_model, hf=True)
        return _model
