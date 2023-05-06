#!/bin/bash
WORKSPACE=${1:-"./workspaces/uss"}    # Default workspace directory

for SPLIT in "balanced_train" "test"
do
    CUDA_VISIBLE_DEVICES=1 python3 uss/dataset_creation/create_audioset_evaluation_meta.py create_evaluation_meta \
        --workspace=$WORKSPACE \
        --split=$SPLIT \
        --output_audios_dir="${WORKSPACE}/evaluation/audioset/2s_segments_${SPLIT}" \
        --output_meta_csv_path="${WORKSPACE}/evaluation/audioset/2s_segments_${SPLIT}.csv"
done
