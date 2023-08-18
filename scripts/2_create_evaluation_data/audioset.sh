#!/bin/bash
WORKSPACE=${1:-"./workspaces/uss"}    # Default workspace directory

DEVICE="cuda"

# 2-second balanced train data.
SPLIT="balanced_train"
CUDA_VISIBLE_DEVICES=0 python evaluation/dataset_creation/audioset.py \
    --indexes_hdf5_path="${WORKSPACE}/hdf5s/indexes/${SPLIT}.h5" \
    --output_audios_dir="${WORKSPACE}/evaluation_data/audioset/2s_segments_${SPLIT}" \
    --output_meta_csv_path="${WORKSPACE}/evaluation_data/audioset/2s_segments_${SPLIT}.csv" \
    --device=$DEVICE

# 2-second test data.
SPLIT="test"
CUDA_VISIBLE_DEVICES=0 python evaluation/dataset_creation/audioset.py \
    --indexes_hdf5_path="${WORKSPACE}/hdf5s/indexes/eval.h5" \
    --output_audios_dir="${WORKSPACE}/evaluation_data/audioset/2s_segments_${SPLIT}" \
    --output_meta_csv_path="${WORKSPACE}/evaluation_data/audioset/2s_segments_${SPLIT}.csv" \
    --device=$DEVICE
