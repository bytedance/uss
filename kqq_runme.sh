
WORKSPACE="/home/tiger/workspaces/casa"

./scripts/0_download_panns_checkpoints.sh

./scripts/2_create_indexes.sh /home/tiger/workspaces/casa/

for SPLIT in "balanced_train" "test"
do
    CUDA_VISIBLE_DEVICES=2 python3 casa/dataset_creation/create_audioset_evaluation_meta.py create_evaluation_meta \
        --workspace=$WORKSPACE \
        --split=$SPLIT \
        --output_audios_dir="${WORKSPACE}/evaluation/audioset/mixtures_sources_${SPLIT}" \
        --output_meta_csv_path="${WORKSPACE}/evaluation/audioset/mixtures_sources_${SPLIT}.csv"
done

CUDA_VISIBLE_DEVICES=1 python3 casa/train.py \
    --workspace=$WORKSPACE \
    --config_yaml="./scripts/train/01.yaml"

CUDA_VISIBLE_DEVICES=0 python3 casa/evaluate.py