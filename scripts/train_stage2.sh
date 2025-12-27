#!/usr/bin/env bash

export TOKENIZERS_PARALLELISM=true
GPUS=${GPUS:-0,1,2,3}
NPROC=${NPROC:-4}
PORT=${PORT:-12345}

CFG=${CFG:-configs/train_mome_stage2.yaml}

CUDA_VISIBLE_DEVICES=${GPUS} torchrun --master_port ${PORT} --nproc_per_node=${NPROC} train.py --cfg-path ${CFG}

