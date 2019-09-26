# AudioSet source separation

## Download 
## 1. Download dataset
Users need to download the balanced and evaluation subset from AudioSet and pack them to hdf5 file. Please follow https://github.com/qiuqiangkong/audioset_tagging_cnn to download the balanced and evaluation subset and pack them to hdf5 file.  
<pre>

dataset_root
├── audios
│    ├── balanced_train_segments
│    |    └── ... (~20550 wavs, the number can be different because some links are missing)
│    └── eval_segments
│         └── ... (~18887 wavs)
└── metadata
     ├── balanced_train_segments.csv
     ├── class_labels_indices.csv
     └── eval_segments.csv
</pre>

## 2. Pack data to hdf5

<pre>
workspace
└── hdf5s
     ├── targets (2.3 GB)
     |    ├── balanced_train.h5
     |    └── eval.h5
     └── waveforms (1.1 TB)
          ├── balanced_train.h5
          ├── eval.h5
</pre>