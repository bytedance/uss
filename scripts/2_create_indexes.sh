#!/bin/bash
WORKSPACE=${1:-"./workspaces/casa"}    # Default workspace directory

# Create balanced eval indexes.
python3 casa/dataset_creation/create_audioset_indexes.py create_indexes \
    --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/eval.h5" \
    --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/eval.h5"

# Create balanced training indexes.
python3 casa/dataset_creation/create_audioset_indexes.py create_indexes \
    --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/balanced_train.h5" \
    --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/balanced_train.h5"

# Create unbalanced training indexes.
for IDX in {00..40}; do
    echo $IDX
    python3 casa/dataset_creation/create_audioset_indexes.py create_indexes \
        --waveforms_hdf5_path=$WORKSPACE"/hdf5s/waveforms/unbalanced_train/unbalanced_train_part$IDX.h5" \
        --indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/unbalanced_train/unbalanced_train_part$IDX.h5"
done

# Combine balanced and unbalanced training indexes to a full training indexes hdf5.
python3 casa/dataset_creation/create_audioset_indexes.py combine_full_indexes \
    --indexes_hdf5s_dir=$WORKSPACE"/hdf5s/indexes" \
    --full_indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/full_train.h5"
