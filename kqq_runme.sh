
WORKSPACE="/home/tiger/workspaces/casa"

./scripts/0_download_panns_checkpoints.sh

./scripts/2_create_indexes.sh /home/tiger/workspaces/casa/

python3 casa/train.py \
    --workspace=$WORKSPACE \
    --config_yaml="./scripts/train/01.yaml"
