#!bin/bash

# Pretrained audio tagging system path
WORKSPACE="/home/tiger/workspaces/audioset_source_separation"
AT_CHECKPOINT_PATH="/home/tiger/released_models/sed/Cnn14_DecisionLevelMax_mAP=0.385.pth"

# Train
CUDA_VISIBLE_DEVICES=2 python3 pytorch/ss_main.py train --workspace=$WORKSPACE --at_checkpoint_path=$AT_CHECKPOINT_PATH --data_type='balanced_train' --model_type='UNet' --loss_type='mae' --balanced='balanced' --augmentation='none' --batch_size=12 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000001 --cuda

# Validate
python3 pytorch/ss_main.py validate --workspace=$WORKSPACE --at_checkpoint_path=$AT_CHECKPOINT_PATH --data_type='balanced_train' --model_type=UNet --condition_type='soft_hard' --loss_type='mae' --batch_size=12 --iteration=200000 --cuda

# Inference and separate a new wav
python3 pytorch/ss_main.py inference_new --workspace=$WORKSPACE --at_checkpoint_path=$AT_CHECKPOINT_PATH --data_type='balanced_train' --model_type=UNet --condition_type='soft_hard' --loss_type='mae' --batch_size=12 --iteration=200000 --cuda

# Plot statistics
python3 utils/plot_statistics.py plot --workspace=$WORKSPACE --select=1
python3 utils/plot_statistics.py plot_waveform_sed --workspace=$WORKSPACE --at_checkpoint_path=$AT_CHECKPOINT_PATH



