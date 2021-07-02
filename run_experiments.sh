#!/bin/bash
# Allowable settings for gpu:n range from 1 to 4. If your application is entirely GPU driven,
# then you do not need to explicilty request cores as one CPU core will be assigned by
# default to act as a master to launch the GPU based calculation. If however your application
# is mixed CPU-GPU then you will need to request the number of cores with --ntasks as is
# required by your job.
# Place your SLURM batch settings here.
# --------------------------------------------------
#SBATCH -J lowgpu-session
#SBATCH -o /homes/xwang/work/slurm/job_name-%x.job_number-%j.nodes-%N.out
#SBATCH -e /homes/xwang/work/slurm/job_name-%x.job_number-%j.nodes-%N.err
#SBATCH -p lowgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
######### SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
######### FOO SBATCH --nodelist highgpu-pg0-1
#SBATCH --exclusive
#
# Resource settings
# --------------------------------------------------
# --------------------------------------------------
# Place your job/script settings here.
# --------------------------------------------------
# Job Commands go here
# --------------------------------------------------
conda activate py39_dm
cd ~/proj/compression

# for MODEL_NAME in resnet18
# do
#     python scripts/run_imagenet.py --model=$MODEL_NAME --config=configs/corsair_cnns_0.yaml
#     python scripts/run_imagenet.py --model=$MODEL_NAME --config=configs/corsair_cnns_1.yaml
#     python scripts/run_imagenet.py --model=$MODEL_NAME --config=configs/corsair_cnns_2.yaml
#     python scripts/run_imagenet.py --model=$MODEL_NAME --config=configs/corsair_cnns_3.yaml
#     python scripts/run_imagenet.py --model=$MODEL_NAME --config=configs/corsair_cnns_4.yaml
# done
# python scripts/run_cifar.py -m resnet20 -c configs/corsair_cnns.yaml -e100

export GLUE_DIR=/tools/d-matrix/ml/data/glue
export GLUE_TASK=MRPC
# export MODEL_NAME=google/bert_uncased_L-2_H-128_A-2 # bert-tiny
# export MODEL_NAME=google/bert_uncased_L-4_H-256_A-4 # bert-mini
# export MODEL_NAME=google/bert_uncased_L-4_H-512_A-8 # bert-small
# export MODEL_NAME=google/bert_uncased_L-8_H-512_A-8 # bert-medium
# export MODEL_NAME=google/bert_uncased_L-12_H-768_A-12 # bert-base
# export MODEL_NAME=bert-base-uncased
export MODEL_NAME=bert-base-cased
# export MODEL_NAME=bert-base-cased-finetuned-mrpc
# export MODEL_NAME=roberta-large-mnli
# export MODEL_NAME=microsoft/deberta-base-mnli
# export MODEL_NAME=microsoft/deberta-large-mnli
# export MODEL_NAME=microsoft/deberta-xlarge-mnli
# export MODEL_NAME=microsoft/deberta-v2-xlarge-mnli
# export MODEL_NAME=microsoft/deberta-v2-xxlarge-mnli

python scripts/run_glue.py \
    --model_name_or_path=$MODEL_NAME \
    --task_name=$GLUE_TASK \
    --max_seq_length=128 \
    --train_file=$GLUE_DIR/$GLUE_TASK/train.tsv \
    --validation_file=$GLUE_DIR/$GLUE_TASK/dev*.tsv \
    --test_file=$GLUE_DIR/$GLUE_TASK/test*.tsv \
    --do_eval \
    --output_dir=./out_dir \
    --overwrite_output_dir \
    --do_train \
    --per_device_train_batch_size=32 \
    --num_train_epochs=8 \
    --learning_rate=5e-5 \
    --lr_scheduler_type=linear \
