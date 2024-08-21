#!/bin/bash
qrsh -g gcd50788 -l rt_G.large=1 -l h_rt=00:30:00 &&
module load python/3.12/3.12.2 &&
module load cuda/12.5/12.5.0 &&
module load nccl/2.21/2.21.5-1 &&
module load cudnn/9.1/9.1.1 &&
cd ~/remote-dir/Marigold_relight &&
source zero123/bin/activate &&
export BASE_DATA_DIR=~/remote-dir/Marigold_relight/data &&
export BASE_CKPT_DIR=~/remote-dir/Marigold_relight/checkpoints &&
accelerate config &&
accelerate launch train.py --config config/train_marigold.yaml --no_wandb   