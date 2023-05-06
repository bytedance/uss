#!/bin/bash

CHECKPOINTS_DIR="${HOME}/.cache/casa"
mkdir -p $CHECKPOINTS_DIR

AT_CHECKPOINT_PATH="${CHECKPOINTS_DIR}/Cnn14_mAP=0.431.pth"
SED_CHECKPOINT_PATH="${CHECKPOINTS_DIR}/Cnn14_DecisionLevelMax_mAP=0.385.pth"

echo "Audio tagging checkpoint path: ${AT_CHECKPOINT_PATH}"
echo "Sound event detection checkpoint path: ${SED_CHECKPOINT_PATH}"

wget -O $AT_CHECKPOINT_PATH "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
wget -O $SED_CHECKPOINT_PATH "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
