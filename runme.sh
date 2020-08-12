#!bin/bash

# Pretrained audio tagging system path
# WORKSPACE="/root/workspaces/audioset_source_separation"
WORKSPACE="/home/tiger/workspaces/audioset_source_separation"
# AT_CHECKPOINT_PATH="/home/tiger/released_models/sed/Cnn14_DecisionLevelMax_mAP=0.385.pth"

# Train
CUDA_VISIBLE_DEVICES=4 python3 pytorch/ss_main.py train --workspace=$WORKSPACE --at_checkpoint_path=$AT_CHECKPOINT_PATH --data_type='balanced_train' --model_type='UNet' --loss_type='mae' --balanced='balanced' --augmentation='none' --mix_type='5' --batch_size=12 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000001 --cuda

CUDA_VISIBLE_DEVICES=2 python3 pytorch/ss_main.py validate --workspace=$WORKSPACE --at_checkpoint_path=$AT_CHECKPOINT_PATH --data_type='balanced_train' --model_type='UNet' --loss_type='mae' --balanced='balanced' --augmentation='none' --batch_size=12 --iteration=100000 --cuda

CUDA_VISIBLE_DEVICES=0 python3 pytorch/ss_main.py inference_new --workspace=$WORKSPACE --at_checkpoint_path=$AT_CHECKPOINT_PATH --data_type='balanced_train' --model_type='UNet' --loss_type='mae' --balanced='balanced' --augmentation='none' --batch_size=2 --iteration=100000 --cuda


python3 pytorch/ss_main.py print

###
CUDA_VISIBLE_DEVICES=5 python3 pytorch/ss_main_16k.py train --workspace=$WORKSPACE --at_checkpoint_path=$AT_CHECKPOINT_PATH --data_type='balanced_train' --model_type='UNet_16k' --loss_type='mae' --balanced='balanced' --augmentation='none' --mix_type='5' --batch_size=12 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000001 --cuda

CUDA_VISIBLE_DEVICES=0 python3 pytorch/ss_main_16k.py inference_new --workspace=$WORKSPACE --at_checkpoint_path=$AT_CHECKPOINT_PATH --data_type='balanced_train' --model_type='UNet_16k' --loss_type='mae' --balanced='balanced' --augmentation='none' --batch_size=2 --iteration=100000 --cuda
