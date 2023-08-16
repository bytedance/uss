#!/bin/bash
WORKSPACE=${1:-"./workspaces/uss"}    # Default workspace directory

DEVICE="cuda"

for SPLIT in "balanced_train" "test"
do
    CUDA_VISIBLE_DEVICES=0 python3 evaluation/dataset_creation/audioset.py \
        --workspace=$WORKSPACE \
        --split=$SPLIT \
        --output_audios_dir="${WORKSPACE}/evaluation/audioset/2s_segments_${SPLIT}" \
        --output_meta_csv_path="${WORKSPACE}/evaluation/audioset/2s_segments_${SPLIT}.csv" \
        --device=$DEVICE
done
