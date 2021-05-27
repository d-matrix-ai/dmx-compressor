from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import os
import random
import sys
import time
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from argparse import Namespace
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
import ctypes
import numpy.ctypeslib as ctl
import numpy as np
import qtorch
from qtorch.quant import Quantizer, fixed_point_quantize, block_quantize, float_quantize
from functools import partial

import corsair

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

logging.getLogger("transformers.modeling_utils").setLevel(
   logging.WARN)  # Reduce logging

# env
load_dotenv(verbose=True)
DATA_PATH = os.getenv("DATA_PATH") or './data_dir'
MODEL_PATH = os.getenv("MODEL_PATH") or './model_dir'
TASK_NAME = 'MRPC'
MODEL_NAME = 'bert-base-uncased'

# numerical
IMC_INPUT_FORMAT = qtorch.BlockFloatingPoint(wl=23, dim=1)
IMC_WEIGHT_FORMAT = qtorch.BlockFloatingPoint(wl=23, dim=1)
IMC_BIAS_FORMAT = qtorch.FloatingPoint(exp=8, man=23)
IMC_OUTPUT_FORMAT = qtorch.FloatingPoint(exp=8, man=23)

SIMD_INPUT_FORMAT = qtorch.FixedPoint(wl=24, fl=4)
SIMD_OUTPUT_FORMAT = qtorch.FixedPoint(wl=24, fl=4)

# components
libc = ctypes.CDLL("./src/modules/libc/actfunction.so")
libc.gelu.argtypes = [ctl.ndpointer(np.float64, flags="aligned, c_contiguous")]
libc.gelu.restype = ctl.ndpointer(
    np.float64, shape=(1000 * 1024,), flags="aligned, c_contiguous"
)


def dmgelu(arr, lid=None):
    mx = torch.max(arr).item()
    inp = torch.reshape(arr, (1000, 1024))
    inp = inp.numpy().astype(np.float64, order="C")
    # print(mx)
    # inp *= (8/mx)
    out = libc.gelu(inp)
    out = torch.Tensor(out)
    # out *= (mx/8)
    # out /= 32
    out = torch.reshape(out, (1000, 1024))
    return out


class DMGELU(nn.Module):
    def __init__(self, bit_scale=0, use_lut=False):
        super().__init__()
        self.qinput = Quantizer(
            forward_number=SIMD_INPUT_FORMAT,
            forward_rounding="nearest",
        )
        self.qoutput = Quantizer(
            forward_number=SIMD_OUTPUT_FORMAT,
            forward_rounding="nearest",
        )
        self.bit_scale = bit_scale
        self.use_lut = use_lut

    def forward(self, x):
        # x = self.qinput(x)
        if self.bit_scale is not None:
            x = fixed_point_quantize(
                x, wl=24, fl=self.bit_scale, symmetric=True, rounding="nearest"
            )
        x = dmgelu(x) if self.use_lut else F.gelu(x)
        if self.bit_scale is not None:
            x = fixed_point_quantize(
                x, wl=24, fl=self.bit_scale, symmetric=True, rounding="nearest"
            )
        # x = self.qoutput(x)
        return x


class DMReLU(nn.Module):
    def __init__(self, bit_scale=0, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.qinput = Quantizer(
            forward_number=SIMD_INPUT_FORMAT,
            forward_rounding="nearest",
        )
        self.qoutput = Quantizer(
            forward_number=SIMD_OUTPUT_FORMAT,
            forward_rounding="nearest",
        )
        self.bit_scale = bit_scale

    def forward(self, x):
        if self.bit_scale is not None:
            # x = self.qinput(x)
            x = fixed_point_quantize(
                x, wl=24, fl=self.bit_scale, symmetric=True, rounding="nearest"
            )
        x = torch.relu_(x) if self.inplace else torch.relu(x)
        if self.bit_scale is not None:
            x = fixed_point_quantize(
                x, wl=24, fl=self.bit_scale, symmetric=True, rounding="nearest"
            )
            # x = self.qoutput(x)
        return x


nn.Linear = corsair.Linear 

# HF transformers
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

# Hard-coded configs
configs = Namespace()
configs.output_dir = os.path.join(MODEL_PATH, "glue", TASK_NAME)
configs.data_dir = os.path.join(DATA_PATH, "glue", TASK_NAME)
configs.model_name_or_path = MODEL_NAME
configs.max_seq_length = 128

# Prepare GLUE task
configs.task_name = TASK_NAME.lower()
configs.processor = processors[configs.task_name]()
configs.output_mode = output_modes[configs.task_name]
configs.label_list = configs.processor.get_labels()
configs.model_type = "bert".lower()
configs.do_lower_case = True

# Set the device, batch size, topology, and caching flags.
configs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configs.per_gpu_eval_batch_size = 8
configs.n_gpu = 0
configs.local_rank = -1
configs.overwrite_cache = False

# Set random seed for reproducibility.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(0)

tokenizer = BertTokenizer.from_pretrained(
    configs.output_dir, do_lower_case=configs.do_lower_case)

model = BertForSequenceClassification.from_pretrained(configs.output_dir)
model.to(configs.device)
print(model)

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                # pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                # pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                # pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def time_model_evaluation(model, configs, tokenizer):
    eval_start_time = time.time()
    result = evaluate(configs, model, tokenizer, prefix="")
    eval_end_time = time.time()
    eval_duration_time = eval_end_time - eval_start_time
    print(result)
    print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))


print("Evaluating model...")
time_model_evaluation(model, configs, tokenizer)

