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

for MODEL_NAME in resnet101
do
    # python scripts/run_imagenet.py --model=$MODEL_NAME --config=configs/corsair_cnns_0.yaml
    # python scripts/run_imagenet.py --model=$MODEL_NAME --config=configs/corsair_cnns_1.yaml
    # python scripts/run_imagenet.py --model=$MODEL_NAME --config=configs/corsair_cnns_2.yaml
    python scripts/run_imagenet.py --model=$MODEL_NAME --config=configs/corsair_cnns_3.yaml
    python scripts/run_imagenet.py --model=$MODEL_NAME --config=configs/corsair_cnns_4.yaml
done
