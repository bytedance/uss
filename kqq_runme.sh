

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

python evaluation/test8.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="fsd50k" \
	--audios_dir="${WORKSPACE}/evaluation/fsd50k/2s_segments_test" \
	--query_embs_dir="${WORKSPACE}/evaluation/embeddings/fsd50k" \
	--device="mps"

python evaluation/test8.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="slakh2100" \
	--audios_dir="${WORKSPACE}/evaluation/slakh2100/2s_segments_test" \
	--query_embs_dir="${WORKSPACE}/evaluation/embeddings/slakh2100" \
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


################# Create data
WORKSPACE="./workspaces/uss"
DATASETS_DIR="./datasets"
DEVICE="cuda"

# AudioSet
for SPLIT in "balanced_train" "test"
do
	CUDA_VISIBLE_DEVICES=0 python3 evaluation/dataset_creation/audioset.py \
	    --indexes_hdf5_path="${WORKSPACE}/hdf5s/indexes/${SPLIT}.h5" \
	    --output_audios_dir="${WORKSPACE}/evaluation/audioset/2s_segments_${SPLIT}" \
	    --output_meta_csv_path="${WORKSPACE}/evaluation/audioset/2s_segments_${SPLIT}.csv" \
	    --device=$DEVICE
done

# FSDKaggle2018
DATASET_TYPE="fsdkaggle2018"

for SPLIT in "train" "test"
do
	python evaluation/dataset_creation/fsd50k.py \
		--dataset_dir="${DATASETS_DIR}/${DATASET_TYPE}" \
		--split=$SPLIT \
		--output_audios_dir="${WORKSPACE}/evaluation/${DATASET_TYPE}/2s_segments_${SPLIT}" \
		--output_meta_csv_path="${WORKSPACE}/evaluation/${DATASET_TYPE}/2s_segments_${SPLIT}.csv"
done

# FSD50k
DATASET_TYPE="fsd50k"

for SPLIT in "train" "test"
do
	python evaluation/dataset_creation/fsdkaggle2018.py \
		--dataset_dir="${DATASETS_DIR}/${DATASET_TYPE}" \
		--split=$SPLIT \
		--output_audios_dir="${WORKSPACE}/evaluation/${DATASET_TYPE}/2s_segments_${SPLIT}" \
		--output_meta_csv_path="${WORKSPACE}/evaluation/${DATASET_TYPE}/2s_segments_${SPLIT}.csv"
done
	
# Slakh2100
DATASET_TYPE="slakh2100"

for SPLIT in "train" "test"
do
	python evaluation/dataset_creation/slakh2100.py \
		--dataset_dir="${DATASETS_DIR}/${DATASET_TYPE}" \
		--split=$SPLIT \
		--output_audios_dir="${WORKSPACE}/evaluation/${DATASET_TYPE}/2s_segments_${SPLIT}" \
		--output_meta_csv_path="${WORKSPACE}/evaluation/${DATASET_TYPE}/2s_segments_${SPLIT}.csv"
done


#################### Emb

WORKSPACE="./workspaces/uss"
DATASETS_DIR="./datasets"

CONFIG_YAML="./scripts/train/ss_model=resunet30,querynet=at_soft,data=balanced.yaml"
CHECKPOINT_PATH="${WORKSPACE}/checkpoints/ss_model=resunet30,querynet=at_soft,data=balanced,devices=1/steps=1000000.ckpt"
DEVICE="cuda"

BASE_CONFIG=`basename $CONFIG_YAML .yaml`

# AudioSet
python evaluation/calculate_embeddings.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="audioset" \
	--audios_dir="${WORKSPACE}/evaluation/audioset/2s_segments_balanced_train" \
	--output_embs_dir="${WORKSPACE}/evaluation/embeddings/audioset/${BASE_CONFIG}" \
	--device=$DEVICE

# FSDKaggle2018
python evaluation/calculate_embeddings.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="fsdkaggle2018" \
	--audios_dir="${WORKSPACE}/evaluation/fsdkaggle2018/2s_segments_train" \
	--output_embs_dir="${WORKSPACE}/evaluation/embeddings/fsdkaggle2018/${BASE_CONFIG}" \
	--device=$DEVICE

# FSD50K
python evaluation/calculate_embeddings.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="fsd50k" \
	--audios_dir="${WORKSPACE}/evaluation/fsd50k/2s_segments_train" \
	--output_embs_dir="${WORKSPACE}/evaluation/embeddings/fsd50k/${BASE_CONFIG}" \
	--device=$DEVICE

# Slakh2100
python evaluation/calculate_embeddings.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="slakh2100" \
	--audios_dir="${WORKSPACE}/evaluation/slakh2100/2s_segments_train" \
	--output_embs_dir="${WORKSPACE}/evaluation/embeddings/slakh2100/${BASE_CONFIG}" \
	--device=$DEVICE

# Musdb18
python evaluation/calculate_embeddings.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="musdb18" \
	--audios_dir="${DATASETS_DIR}/musdb18hq/train" \
	--output_embs_dir="${WORKSPACE}/evaluation/embeddings/musdb18" \
	--device=$DEVICE

# Voicebank-Demand
python evaluation/calculate_embeddings.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type="voicebank-demand" \
	--audios_dir="${DATASETS_DIR}/voicebank-demand" \
	--output_embs_dir="${WORKSPACE}/evaluation/embeddings/voicebank-demand" \
	--device=$DEVICE

#################### Evaluate

DATASET_TYPE="audioset"
SPLIT="test"

# Audioset
python evaluation/separate_and_evaluate.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type=$DATASET_TYPE \
	--audios_dir="${WORKSPACE}/evaluation/${DATASET_TYPE}/2s_segments_${SPLIT}" \
	--query_embs_dir="${WORKSPACE}/evaluation/embeddings/${DATASET_TYPE}/${BASE_CONFIG}" \
	--device=$DEVICE

# Fsdkaggle2018
DATASET_TYPE="fsdkaggle2018"

python evaluation/separate_and_evaluate.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type=$DATASET_TYPE \
	--audios_dir="${WORKSPACE}/evaluation/${DATASET_TYPE}/2s_segments_${SPLIT}" \
	--query_embs_dir="${WORKSPACE}/evaluation/embeddings/${DATASET_TYPE}/${BASE_CONFIG}" \
	--device=$DEVICE

# FSD50k
DATASET_TYPE="fsd50k"

python evaluation/separate_and_evaluate.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type=$DATASET_TYPE \
	--audios_dir="${WORKSPACE}/evaluation/${DATASET_TYPE}/2s_segments_${SPLIT}" \
	--query_embs_dir="${WORKSPACE}/evaluation/embeddings/${DATASET_TYPE}/${BASE_CONFIG}" \
	--device=$DEVICE

# Slakh2100
DATASET_TYPE="slakh2100"

python evaluation/separate_and_evaluate.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type=$DATASET_TYPE \
	--audios_dir="${WORKSPACE}/evaluation/${DATASET_TYPE}/2s_segments_${SPLIT}" \
	--query_embs_dir="${WORKSPACE}/evaluation/embeddings/${DATASET_TYPE}/${BASE_CONFIG}" \
	--device=$DEVICE

# Musdb18
DATASET_TYPE="musdb18"

python evaluation/separate_and_evaluate.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type=$DATASET_TYPE \
	--audios_dir="${DATASETS_DIR}/musdb18hq/${SPLIT}" \
	--query_embs_dir="${WORKSPACE}/evaluation/embeddings/${DATASET_TYPE}/${BASE_CONFIG}" \
	--device=$DEVICE

# Voicebank-Demand
DATASET_TYPE="voicebank-demand"

python evaluation/separate_and_evaluate.py \
	--config_yaml=$CONFIG_YAML \
	--checkpoint_path=$CHECKPOINT_PATH \
	--dataset_type=$DATASET_TYPE \
	--audios_dir="${DATASETS_DIR}/${DATASET_TYPE}" \
	--query_embs_dir="${WORKSPACE}/evaluation/embeddings/${DATASET_TYPE}/${BASE_CONFIG}" \
	--device=$DEVICE
