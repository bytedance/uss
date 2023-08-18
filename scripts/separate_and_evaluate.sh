#!/bin/bash

WORKSPACE="./workspaces/uss"

CONFIG_YAML="./scripts/train_configs/ss_model=resunet30,querynet=at_soft,data=balanced.yaml"
CHECKPOINT_PATH="checkpoints/ss_model=resunet30,querynet=at_soft,data=full,devices=8,steps=1000000.ckpt"

BASE_CONFIG=`basename $CHECKPOINT_PATH .yaml`

DEVICE="cuda"
SPLIT="test"

# Evaluate on all datasets. Users may evaluate the datasets in individual terminals for speed up.
for DATASET_TYPE in "audioset" "fsdkaggle2018" "fsd50k" "slakh2100" "musdb18" "voicebank-demand"
do
	CUDA_VISIBLE_DEVICES=0 python evaluation/separate_and_evaluate.py \
		--config_yaml=$CONFIG_YAML \
		--checkpoint_path=$CHECKPOINT_PATH \
		--dataset_type=$DATASET_TYPE \
		--audios_dir="${WORKSPACE}/evaluation/${DATASET_TYPE}/2s_segments_${SPLIT}" \
		--query_embs_dir="${WORKSPACE}/evaluation/embeddings/${DATASET_TYPE}/${BASE_CONFIG}" \
		--device=$DEVICE
done
