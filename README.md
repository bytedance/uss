# Computation Auditory Scene Analysis (CASA) and Universal Source Separation with Weakly labelled Data

This is the PyTorch implementation of the Universal Source Separation with Weakly labelled Data [1]. The CASA system is able to automatically detect and separate up to hundreds of sound classes using a single model. 

## 1. Installation
```bash
pip install casa
```

## 2. Usage

### Download test audio (optional)
```bash
wget -O "harry_potter.flac" "https://sandbox.zenodo.org/record/1196560/files/harry_potter.flac?download=1"
```

### Default: automatic detect and separate
```bash
casa -i "harry_potter.flac"
```

### Separate with different AudioSet hierarchy levels (The same as default)
```bash
casa -i "harry_potter.flac" --levels 1 2 3
```

### Separate by class IDs
```bash
casa -i "harry_potter.flac" --class_ids 0 1 2 3 4
```

### Separate by queries

Download query audios (optional)
```bash
wget -O "queries.zip" "https://sandbox.zenodo.org/record/1196562/files/queries.zip?download=1"
unzip queries.zip
```

```bash
casa -i "harry_potter.flac" --queries_dir "queries/speech"
```

## 3. Inference by cloning the repo

Developers could also git clone this repo and run the inference in the repo. This will let users to have more flexibility to modify the inference code.

### Set up environment

```bash
conda create -n casa python=3.8
conda activate casa
pip install -r requirements.txt
```

### Inference

```python
CUDA_VISIBLE_DEVICES=0 python3 casa/inference.py \
    --audio_path=./resources/harry_potter.flac \
    --levels 1 2 3 \
    --config_yaml="./scripts/train/ss_model=resunet30,querynet=at_soft,data=full.yaml" \
    --checkpoint_path=""
```

## Train from Scratch

## Download dataset. 

The [scripts/1_download_dataset.sh](scripts/1_download_dataset.sh) script is used for downloading all audio and metadata from the internet. The total size of AudioSet is around 1.1 TB. Notice there can be missing files on YouTube, so the numebr of files downloaded by users can be different from time to time. Our downloaded version contains 20550 / 22160 of the balaned training subset, 1913637 / 2041789 of the unbalanced training subset, and 18887 / 20371 of the evaluation subset. 

For reproducibility, our downloaded dataset can be accessed at: link: [https://pan.baidu.com/s/13WnzI1XDSvqXZQTS-Kqujg](https://pan.baidu.com/s/13WnzI1XDSvqXZQTS-Kqujg), password: 0vc2

The downloaded data looks like:

<pre>

dataset_root
├── audios
│    ├── balanced_train_segments
│    |    └── ... (~20550 wavs, the number can be different from time to time)
│    ├── eval_segments
│    |    └── ... (~18887 wavs)
│    └── unbalanced_train_segments
│         ├── unbalanced_train_segments_part00
│         |    └── ... (~46940 wavs)
│         ...
│         └── unbalanced_train_segments_part40
│              └── ... (~39137 wavs)
└── metadata
     ├── balanced_train_segments.csv
     ├── class_labels_indices.csv
     ├── eval_segments.csv
     ├── qa_true_counts.csv
     └── unbalanced_train_segments.csv
</pre>

## 2. Pack waveforms into hdf5 files
The [scripts/2_pack_waveforms_to_hdf5s.sh](scripts/2_pack_waveforms_to_hdf5s.sh) script is used for packing all raw waveforms into 43 large hdf5 files for speed up training: one for balanced training subset, one for evaluation subset and 41 for unbalanced traning subset. The packed files looks like:

<pre>
workspace
└── hdf5s
     ├── targets (2.3 GB)
     |    ├── balanced_train.h5
     |    ├── eval.h5
     |    └── unbalanced_train
     |        ├── unbalanced_train_part00.h5
     |        ...
     |        └── unbalanced_train_part40.h5
     └── waveforms (1.1 TB)
          ├── balanced_train.h5
          ├── eval.h5
          └── unbalanced_train
              ├── unbalanced_train_part00.h5
              ...
              └── unbalanced_train_part40.h5
</pre>

```bash
./scripts/train.sh
```

## Reference