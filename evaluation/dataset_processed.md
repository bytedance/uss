
## Processed Dataset for Evaluation

Users can download the processed evaluation data directly. The proceessed datasets for evaluation include the AudioSet, FSDKaggle2018, FSD50k, Slakh2100, MUSDB18, and Voicebank-Demand datasets. The AudioSet, FSDKaggle2018, FSD50k, and Slakh2100 datasets are processed into 2-second segments. The MUSDB18 and Voicebank-Demand datasets remain their durations for fair comparison with previous works.

Here is a list of processed datasets. Please ensure the datasets are completely downloaded and have the following tree structure.

<pre>
datasets
├── audioset
├── fsdkaggle2018
├── fsd50k
├── slakh2100
├── musdb18hq
└── voicebank-demand
</pre>

## Audioset Dataset

<pre>
audioset
├── 2s_segments_balanced_train
│    └── 527 folders
├── 2s_segments_balanced_test
│    └── 527 folders
├── 2s_segments_train.csv
└── 2s_segments_test.csv
</pre>

## FSDKaggle2018 Dataset

<pre>
fsdkaggle2018
├── 2s_segments_balanced_train
│    └── 41 folders
├── 2s_segments_balanced_test
│    └── 41 folders
├── 2s_segments_train.csv
└── 2s_segments_test.csv
</pre>

## FSD50k Dataset

<pre>
fsd50k
├── 2s_segments_balanced_train
│    └── 196 folders
├── 2s_segments_balanced_test
│    └── 195 folders
├── 2s_segments_train.csv
└── 2s_segments_test.csv
</pre>

## Slakh2100 dataset

<pre>
slakh2100
├── 2s_segments_balanced_train
│    └── 165 folders
├── 2s_segments_balanced_test
│    └── 148 folders
├── 2s_segments_train.csv
└── 2s_segments_test.csv
</pre>

## MUSDB18 dataset

<pre>
musdb18hq
├── train
│    └── ... (100 songs)
└── test
     └── ... (50 songs)
</pre>

## Voicebank-Demand dataset

<pre>
voicebank-demand
├── clean_trainset_wav
│    └── ... (11572 wavs)
├── noisy_trainset_wav
│    └── ... (11572 wavs)
├── clean_testset_wav
│    └── ... (824 wavs)
└── noisy_testset_wav
     └── ... (824 wavs)
</pre>