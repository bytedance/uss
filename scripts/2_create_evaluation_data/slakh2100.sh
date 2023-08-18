#!/bin/bash
DATASET_DIR=${2:-"./datasets/slakh2100"}    # Dataset directory
WORKSPACE=${2:-"./workspaces/uss"}    # Default workspace directory

DEVICE="cuda"

for SPLIT in "train" "test"
do
    python evaluation/dataset_creation/slakh2100.py \
        --dataset_dir=$DATASET_DIR \
        --split=$SPLIT \
        --output_audios_dir="${WORKSPACE}/evaluation_data/slakh2100/2s_segments_${SPLIT}" \
        --output_meta_csv_path="${WORKSPACE}/evaluation_data/slakh2100/2s_segments_${SPLIT}.csv"
done