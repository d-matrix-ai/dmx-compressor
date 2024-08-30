from typing import Optional, Dict
from transformers import pipeline as hfpipeline
from dmx.compressor.fx.transform import substitute_transform
import transformers
from .model import DmxModelMixin, DmxConfig
import evaluate
from evaluate import evaluator
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from .model import DmxModel, DmxConfig
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM
from accelerate.utils import compute_module_sizes, find_tied_parameters
import re


def get_config_file(repo_name, revision, config_name):
    if config_name in ["BASELINE", "BASIC"]:
        return None
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

            # NOTE: assuming pipe.model is in BASELINE mode
            pipe.model.configure(None, *eval(f"config_rules.{dmx_config_name}"))
        else:
            raise RuntimeError(f"illegal dmx_config: {dmx_config_name}")


def prepare_dataset_and_column(
    dataset,
    column_name=None,
    dataset_version=None,
    dataset_split="test",
    seed=42,
    trust_remote_code=True,
):
    dataset_column_mapping = {
        "wikitext": "text",
        "ptb_text_only": "sentence",
        "lambada": "text",
        "EleutherAI/lambada_openai": "text",
        # Add more datasets and their respective column names here
    }

    if dataset == "squad":
        column_name = None
    elif not column_name and dataset in dataset_column_mapping:
        column_name = dataset_column_mapping[dataset]
    elif not column_name:
        raise ValueError(
            f"Column name not found for dataset '{dataset}'. Please provide the column_name."
        )

    dataset = load_dataset(
        dataset,
        dataset_version,
        split=dataset_split,
        trust_remote_code=trust_remote_code,
    )
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


def eval_question_answering(
    model,
    tokenizer,
    dataset,
    metric=None,
    dataset_split="test",
    revision="main",
    column_name=None,
    dataset_version=None,
):
    dataset, column_name = prepare_dataset_and_column(
        dataset, column_name, dataset_version, dataset_split
    )
    task_evaluator = evaluator("question-answering")
    results = task_evaluator.compute(
        model_or_pipeline=model, tokenizer=tokenizer, data=dataset
    )
    return results


def eval_text_generation(
    model,
    tokenizer,
    dataset,
    metric,
    dataset_split="test",
    revision="main",
    column_name=None,
    dataset_version=None,
):
    dataset, column_name = prepare_dataset_and_column(
        dataset, column_name, dataset_version, dataset_split
    )

    metric = evaluate.load(metric, module_type="metric")

    results = metric.compute(
        model=model,
        tokenizer=tokenizer,
        revision=revision,
        references=dataset[column_name],
    )
    return results


def pipe_eval(
    model,
    tokenizer,
    dataset,
    metric,
    task,
    revision,
    dataset_split="test",
    column_name=None,
    dataset_version=None,
):
    task_eval_mapping = {
        "text-generation": eval_text_generation,
        "question-answering": eval_question_answering,
    }

    if task not in task_eval_mapping:
        raise ValueError(f"Unsupported task type '{task}'.")

    eval_function = task_eval_mapping[task]
    return eval_function(
        model,
        tokenizer,
        dataset,
        metric,
        dataset_split,
        revision,
        column_name,
        dataset_version,
    )

def contains_number(string):
    """
    This function checks whether a string contains a number or not.
    This is used to identify hidden layers in the model.
    """
    return bool(re.search(r"\d", string))


def get_modules(root: torch.nn.Module, prefix: str) -> Dict[str, torch.nn.Module]:
    """
    A function that recursively traverses the model from the given root module and
    returns a dictionary of submodules for device mapping. In accordance with the format
    of device_map = "auto", only submodules that are leaf nodes or hidden layers are
    included in the dictionary; submodules of hidden layers are ignored.

    Args:
        root (torch.nn.Module): model/module to traverse
        prefix (str): prefix for the submodule names

    Returns:
        Dict[str, torch.nn.Module]: dictionary of submodules
    """
    modules = {}
    for name, module in root.named_children():
        if len(list(module.children())) == 0 or contains_number(name):
            modules[prefix + name] = module
        else:
            submodules = get_modules(module, prefix + name + ".")
            modules.update(submodules)
    return modules


def balanced_device_map(model: str, revision: Optional[str] = "main") -> Dict[str, int]:
    """
    A function that computes a custom device map for the given model that distributes model weights
    evenly across all devices. Enable with device_map = "balanced" when calling pipeline.

    Args:
        model (str): model name on huggingface
        revision (str): revision of the model on huggingface

    Returns:
        Dict[str, int]: dictionary of the device map
    """
    model = AutoModelForCausalLM.from_pretrained(
        model, revision=revision, device_map="meta"
    )
    module_sizes = compute_module_sizes(model)
    tied_parameters = find_tied_parameters(model)
    modules = get_modules(model, "")

    params = {}
    for k, v in modules.items():
        params[k] = module_sizes[k]

    device_map = {}
    total_params = sum(params.values())
    num_devices = torch.cuda.device_count()

    cur_device = 0
    params_on_cur_device = 0
    accumulated_params = 0
    average_params = total_params // num_devices

    # place all embedding modules together on the first device
    for name, module in modules.items():
        if isinstance(module, torch.nn.Embedding):
            device_map[name] = cur_device
            params_on_cur_device += params[name]
            continue

    # distribute remaining modules across all devices
    for name, module in modules.items():
        if isinstance(module, torch.nn.Embedding):
            continue
        if params_on_cur_device > average_params or (
            params_on_cur_device != 0
            and params_on_cur_device + params[name] > average_params * 1.2
        ):
            cur_device = min(cur_device + 1, num_devices - 1)
            accumulated_params += params_on_cur_device
            params_on_cur_device = 0
            average_params = (total_params - accumulated_params) // (
                num_devices - cur_device
            )
        device_map[name] = cur_device
        params_on_cur_device += params[name]

    for pair in tied_parameters:
        device_map[pair[0][: -len(".weight")]] = device_map[pair[1][: -len(".weight")]]

    return device_map


def pipeline(
    *args,
    dmx_config: Optional[str] = None,
    trust_remote_code: bool = True,
    device_map: Optional[str] = "auto",
    **kwargs,
):
    if device_map == "balanced":
        device_map = balanced_device_map(
            kwargs.get("model"), kwargs.get("revision", "main")
        )
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
        pipe.model
    )
    pipe.evaluate = lambda metric, dataset, column_name=None, dataset_version=None, dataset_split="test": pipe_eval(
        pipe.model,
        pipe.tokenizer,
        dataset,
        metric,
        pipe.task,
        pipe.revision,
        dataset_split,
        column_name,
        dataset_version,
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
