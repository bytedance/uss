#!/bin/bash
DATASET_DIR=${2:-"./datasets/fsdkaggle2018"}    # Dataset directory
WORKSPACE=${2:-"./workspaces/uss"}    # Default workspace directory

DEVICE="cuda"

for SPLIT in "train" "test"
do
    python evaluation/dataset_creation/fsdkaggle2018.py \
        --dataset_dir=$DATASET_DIR \
        --split=$SPLIT \
        --output_audios_dir="${WORKSPACE}/evaluation_data/fsdkaggle2018/2s_segments_${SPLIT}" \
        --output_meta_csv_path="${WORKSPACE}/evaluation_data/fsdkaggle2018/2s_segments_${SPLIT}.csv"
done