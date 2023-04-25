#!/bin/bash

pip install -r requirements.txt

# Download checkpoints
mkdir -p "./downloaded_checkpoints"

hdfs dfs -get "hdfs://haruna/home/byte_speech_sv/user/kongqiuqiang/workspaces/casa/checkpoints/ss_model=resunet30,querynet=at_soft,full,devices=8/step=100000.ckpt" "./downloaded_checkpoints/ss_model=resunet30,querynet=at_soft,full,devices=8,step=100000.ckpt"

# Separate
CUDA_VISIBLE_DEVICES=0 python3 casa/inference.py \
    --audio_path=./resources/harry_potter.flac \
    --levels 1 2 3

CUDA_VISIBLE_DEVICES=0 python3 casa/inference.py \
    --audio_path=./resources/harry_potter.flac \
    --class_ids 0 1 2

CUDA_VISIBLE_DEVICES=0 python3 casa/inference.py \
    --audio_path=./resources/harry_potter.flac \
    --queries_dir="./resources/queries"
