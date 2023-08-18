#!/bin/bash

WORKSPACE="./workspaces/uss"

CONFIG_YAML="./scripts/train_configs/ss_model=resunet30,querynet=emb,data=balanced.yaml"
CHECKPOINT_PATH="checkpoints/ss_model=resunet30,querynet=emb,data=balanced,devices=1,steps=1000000.ckpt"

BASE_CONFIG=`basename $CHECKPOINT_PATH .ckpt`

DEVICE="cuda"
SPLIT="test"

# Users may evaluate the datasets in individual terminals for speed up.
for DATASET_TYPE in "audioset" "fsdkaggle2018" "fsd50k" "slakh2100" "musdb18" "voicebank-demand"
do
	CUDA_VISIBLE_DEVICES=0 python evaluation/separate_and_evaluate.py \
		--config_yaml=$CONFIG_YAML \
		--checkpoint_path=$CHECKPOINT_PATH \
		--dataset_type=$DATASET_TYPE \
		--audios_dir="${WORKSPACE}/evaluation_data/${DATASET_TYPE}/2s_segments_${SPLIT}" \
		--query_embs_dir="${WORKSPACE}/evaluation_embeddings/${DATASET_TYPE}/${BASE_CONFIG}" \
		--metrics_path="${WORKSPACE}/evaluation_metrics/${DATASET_TYPE}/${BASE_CONFIG}" \
		--device=$DEVICE
done
