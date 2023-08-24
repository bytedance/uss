#!/bin/bash
WORKSPACE="./workspaces/uss"    # Default workspace directory

CONFIG_YAML="./scripts/train_configs/ss_model=resunet30,querynet=emb,data=full.yaml"
CHECKPOINT_PATH="checkpoints/ss_model=resunet30,querynet=emb,data=balanced,devices=1,steps=1000000.ckpt"

BASE_CONFIG=`basename $CHECKPOINT_PATH .ckpt`

DEVICE="cuda"

# AudioSet
CUDA_VISIBLE_DEVICES=0 python evaluation/calculate_embeddings.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="audioset" \
	--audios_dir="${WORKSPACE}/evaluation_data/audioset/2s_segments_balanced_train" \
	--output_embs_dir="${WORKSPACE}/evaluation_embeddings/audioset/${BASE_CONFIG}" \
	--device=$DEVICE

# FSDKaggle2018, FSD50k, and Slakh2100 datasets
for DATASET_TYPE in "fsdkaggle2018" "fsd50k" "slakh2100"
do
	python evaluation/calculate_embeddings.py \
		--config_yaml=$CONFIG_YAML \
		--checkpoint_path=$CHECKPOINT_PATH \
		--dataset_type=$DATASET_TYPE \
		--audios_dir="${WORKSPACE}/evaluation_data/${DATASET_TYPE}/2s_segments_train" \
		--output_embs_dir="${WORKSPACE}/evaluation_embeddings/${DATASET_TYPE}/${BASE_CONFIG}" \
		--device=$DEVICE
done

# Musdb18
python evaluation/calculate_embeddings.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="musdb18" \
	--audios_dir="datasets/musdb18hq/train" \
	--output_embs_dir="${WORKSPACE}/evaluation_embeddings/${DATASET_TYPE}/${BASE_CONFIG}" \
	--device=$DEVICE

# Voicebank-Demand
python evaluation/calculate_embeddings.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="voicebank-demand" \
	--audios_dir="datasets/voicebank-demand" \
	--output_embs_dir="${WORKSPACE}/evaluation_embeddings/${DATASET_TYPE}/${BASE_CONFIG}" \
	--device=$DEVICE
