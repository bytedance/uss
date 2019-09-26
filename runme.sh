#!bin/bash

WORKSPACE="/mnt/cephfs_new_wj/speechsv/qiuqiang.kong/workspaces/pub_audioset_tagging_cnn_transfer"
AT_CHECKPOINT_PATH="/mnt/cephfs_new_wj/speechsv/qiuqiang.kong/workspaces/pub_audioset_tagging_cnn_transfer/checkpoints/main/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13_DecisionLevelMax/loss_type=clip_bce/balanced=balanced/augmentation=none/batch_size=32/260000_iterations.pth"

python3 pytorch/ss_main.py train --workspace=$WORKSPACE --at_checkpoint_path=$AT_CHECKPOINT_PATH --data_type='balanced_train' --model_type=UNet --condition_type='soft_hard' --wiener_filter --loss_type='mae' --batch_size=12 --learning_rate=1e-3 --resume_iteration=0 --early_stop=1000001 --cuda

python3 utils/plot_statistics.py plot --workspace=$WORKSPACE --select=1
python3 utils/plot_statistics.py plot_waveform_sed --workspace=$WORKSPACE --at_checkpoint_path=$AT_CHECKPOINT_PATH

python3 pytorch/ss_main.py validate --workspace=$WORKSPACE --at_checkpoint_path=$AT_CHECKPOINT_PATH --data_type='balanced_train' --model_type=UNet --condition_type='soft' --loss_type='mae' --batch_size=12 --iteration=200000 --cuda

python3 pytorch/ss_main.py inference_new --workspace=$WORKSPACE --at_checkpoint_path=$AT_CHECKPOINT_PATH --data_type='balanced_train' --model_type=UNet --condition_type='soft' --loss_type='mae' --batch_size=12 --iteration=200000 --cuda