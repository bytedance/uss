

WORKSPACE="/Users/bytedance/workspaces/uss"

### FSDKaggle2018
DATASET_DIR="/Users/bytedance/datasets/fsdkaggle2018"
SPLIT="train"
OUTPUT_AUDIOS_DIR="${WORKSPACE}/evaluation/fsdkaggle2018/2s_segments_${SPLIT}"
OUTPUT_META_CSV_PATH="${WORKSPACE}/evaluation/fsdkaggle2018/2s_segments_${SPLIT}.csv"

python ./evaluation/fsdkaggle2018/create_evaluation_data.py \
	--dataset_dir=$DATASET_DIR \
	--split=$SPLIT \
	--output_audios_dir=$OUTPUT_AUDIOS_DIR \
	--output_meta_csv_path=$OUTPUT_META_CSV_PATH
	

### FSD50k
DATASET_DIR="/Users/bytedance/datasets/fsd50k"
SPLIT="train"
OUTPUT_AUDIOS_DIR="${WORKSPACE}/evaluation/fsd50k/2s_segments_${SPLIT}"
OUTPUT_META_CSV_PATH="${WORKSPACE}/evaluation/fsd50k/2s_segments_${SPLIT}.csv"

python ./evaluation/fsd50k/create_evaluation_data.py \
	--dataset_dir=$DATASET_DIR \
	--split=$SPLIT \
	--output_audios_dir=$OUTPUT_AUDIOS_DIR \
	--output_meta_csv_path=$OUTPUT_META_CSV_PATH

### Slakh2100
DATASET_DIR="/Users/bytedance/datasets/slakh2100"
SPLIT="train"
OUTPUT_AUDIOS_DIR="${WORKSPACE}/evaluation/slakh2100/2s_segments_${SPLIT}"
OUTPUT_META_CSV_PATH="${WORKSPACE}/evaluation/slakh2100/2s_segments_${SPLIT}.csv"

python ./evaluation/slakh2100/create_evaluation_data.py \
	--dataset_dir=$DATASET_DIR \
	--split=$SPLIT \
	--output_audios_dir=$OUTPUT_AUDIOS_DIR \
	--output_meta_csv_path=$OUTPUT_META_CSV_PATH

######### Calcualte emb

CONFIG_YAML="./scripts/train/ss_model=resunet30,querynet=at_soft,data=balanced.yaml"
CHECKPOINT_PATH="${WORKSPACE}/checkpoints/ss_model=resunet30,querynet=at_soft,data=balanced,devices=1/steps=1000000.ckpt"

### FSDKaggle2018
python uss/test9.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="fsdkaggle2018" \
	--audios_dir="${WORKSPACE}/evaluation/fsdkaggle2018/2s_segments_train" \
	--output_embs_dir="${WORKSPACE}/evaluation/embeddings/fsdkaggle2018" \
	--device="mps"

# FSD50k
python uss/test9.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="fsd50k" \
	--audios_dir="${WORKSPACE}/evaluation/fsd50k/2s_segments_train" \
	--output_embs_dir="${WORKSPACE}/evaluation/embeddings/fsd50k" \
	--device="mps"

# Voicebank-Demand
python uss/test9.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="voicebank-demand" \
	--audios_dir="/Users/bytedance/datasets/voicebank-demand" \
	--output_embs_dir="${WORKSPACE}/evaluation/embeddings/voicebank-demand" \
	--device="mps"

python uss/test9.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="voicebank-demand" \
	--audios_dir="/Users/bytedance/datasets/voicebank-demand" \
	--output_embs_dir="${WORKSPACE}/evaluation/embeddings/voicebank-demand" \
	--device="mps"

python uss/test9.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="musdb18" \
	--audios_dir="/Users/bytedance/datasets/musdb18hq/train" \
	--output_embs_dir="${WORKSPACE}/evaluation/embeddings/musdb18" \
	--device="mps"

python uss/test9.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="slakh2100" \
	--audios_dir="${WORKSPACE}/evaluation/slakh2100/2s_segments_train" \
	--output_embs_dir="${WORKSPACE}/evaluation/embeddings/slakh2100" \
	--device="mps"

###### Evaluate
# Fsdkaggle2018
python evaluation/test8.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="fsdkaggle2018" \
	--audios_dir="${WORKSPACE}/evaluation/fsdkaggle2018/2s_segments_test" \
	--query_embs_dir="${WORKSPACE}/evaluation/embeddings/fsdkaggle2018" \
	--device="mps"

# voicebank-demand
python evaluation/test8.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="voicebank-demand" \
	--audios_dir="/Users/bytedance/datasets/voicebank-demand/" \
	--query_embs_dir="${WORKSPACE}/evaluation/embeddings/voicebank-demand" \
	--device="mps"


# musdb18
python evaluation/test8.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="musdb18" \
	--audios_dir="/Users/bytedance/datasets/musdb18hq/test" \
	--query_embs_dir="${WORKSPACE}/evaluation/embeddings/musdb18" \
	--device="mps"
