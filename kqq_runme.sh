
WORKSPACE="/home/tiger/workspaces/casa"
WORKSPACE="./workspaces/casa"

./scripts/0_download_checkpoints.sh

./scripts/1_download_audioset.sh

./scripts/1_pack_waveforms_to_hdf5s.sh

./scripts/2_create_indexes.sh /home/tiger/workspaces/casa/

for SPLIT in "balanced_train" "test"
do
    CUDA_VISIBLE_DEVICES=1 python3 casa/dataset_creation/create_audioset_evaluation_meta.py create_evaluation_meta \
        --workspace=$WORKSPACE \
        --split=$SPLIT \
        --output_audios_dir="${WORKSPACE}/evaluation/audioset/2s_segments_${SPLIT}" \
        --output_meta_csv_path="${WORKSPACE}/evaluation/audioset/2s_segments_${SPLIT}.csv"
done

# # Combine balanced and unbalanced training indexes to a full training indexes hdf5
# python3 casa/dataset_creation/create_indexes.py combine_full_indexes \
#     --indexes_hdf5s_dir=$WORKSPACE"/hdf5s/indexes" \
#     --full_indexes_hdf5_path=$WORKSPACE"/hdf5s/indexes/full_train.h5"

CUDA_VISIBLE_DEVICES=1 python3 casa/train.py \
    --workspace=$WORKSPACE \
    --config_yaml="./scripts/train/01a.yaml"

CUDA_VISIBLE_DEVICES=0 python3 casa/evaluate.py

python3 casa/plot.py


CUDA_VISIBLE_DEVICES=3 python3 casa/inference.py \
    --audio_path=./resources/harry_potter.flac \
    --levels 1 2 3 \
    --config_yaml="./scripts/train/ss_model=resunet30,querynet=at_soft,data=full.yaml" \
    --checkpoint_path="/home/tiger/workspaces/casa/checkpoints/train/ss_model=resunet30,querynet=at_soft,data=full,devices=8/step=1000000.ckpt"

CUDA_VISIBLE_DEVICES=3 python3 casa/inference.py \
    --audio_path=./resources/harry_potter.flac \
    --levels 1 2 3 \
    --config_yaml="./scripts/train/ss_model=resunet30,querynet=at_soft,data=full.yaml" \
    --checkpoint_path="/home/tiger/workspaces/casa/checkpoints/train/ss_model=resunet30,querynet=at_soft,data=full,devices=8/step=1000000.ckpt"

### evaluate on musdb18
CUDA_VISIBLE_DEVICES=3 python3 casa/evaluate_musdb18.py calculate_condition \
    --queries_dir="resources/queries/drums" \
    --config_yaml="./scripts/train/ss_model=resunet30,querynet=emb,data=full.yaml" \
    --checkpoint_path="./workspaces/casa/checkpoints/train/config=ss_model=resunet30,querynet=emb,gpus=1,devices=1/step=300000.ckpt"


CUDA_VISIBLE_DEVICES=3 python3 casa/evaluate_musdb18.py evaluate \
    --query_emb_path="query_conditions/config=ss_model=resunet30,querynet=emb,data=full/drums.pkl" \
    --config_yaml="./scripts/train/ss_model=resunet30,querynet=emb,data=full.yaml" \
    --checkpoint_path="./workspaces/casa/checkpoints/train/config=ss_model=resunet30,querynet=emb,gpus=1,devices=1/step=300000.ckpt"
