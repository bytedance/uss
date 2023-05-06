#!/bin/bash
WORKSPACE=${1:-"./workspaces/uss"}    # Default workspace directory

# Train
CUDA_VISIBLE_DEVICES=1 python3 uss/train.py \
    --workspace=$WORKSPACE \
    --config_yaml="./scripts/train/ss_model=resunet30,querynet=at_soft,data=balanced.yaml"
