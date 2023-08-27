
## Origin Dataset for Evaluation

**Users can skip this stage** and directly download the processed evaluation data. In case the users wish to process the evaluation data by themselves, please see details below.

The evaluation datasets include AudioSet, FSDKaggle2018, FSD50k, Slakh2100, MUSDB18, Voicebank-Demand. The AudioSet, FSDKaggle2018, FSD50k, and Slakh2100 datasets are processed into 2-second segments. The MUSDB18 and Voicebank-Demand datasets remain their durations for fair comparison with previous works.

Here is a list of datasets. Please ensure the datasets are completely downloaded and have the following tree structure.

<pre>
datasets
├── audioset [download](https://github.com/qiuqiangkong/audioset_tagging_cnn)
├── fsdkaggle2018 [download](https://zenodo.org/record/2552860)
├── fsd50k [download](https://zenodo.org/record/4060432)
├── slakh2100 [download](https://zenodo.org/record/4599666)
├── musdb18hq [download](https://zenodo.org/record/3338373)
└── voicebank-demand [download](https://datashare.ed.ac.uk/handle/10283/1942?show=full)
</pre>

## Audioset Dataset

<pre>
audioset
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

## FSDKaggle2018 Dataset

<pre>
fsdkaggle2018
├── FSDKaggle2018.audio_train
│    └── ... (9473 wavs)
├── FSDKaggle2018.audio_test
│    └── ... (1600 wavs)
├── FSDKaggle2018.meta
│    ├── train_post_competition.csv
│    └── test_post_competition_scoring_clips.csv
└── FSDKaggle2018.doc
     └── ...
</pre>

## FSD50k Dataset

<pre>
fsd50k
├── FSD50K.dev_audio
│    └── ... (40966 wavs)
├── FSD50K.eval_audio
│    └── ... (10231 wavs)
├── FSD50K.ground_truth
│    ├── dev.csv
│    ├── eval.csv
│    └── vocabulary.csv
├── FSD50K.metadata
│    └── ... 
└── FSDKaggle2018.doc
     └── ...
</pre>

## Slakh2100 dataset

<pre>
slakh2100
├── train
│    └── ... (1500 tracks)
├── validation
│    └── ... (375 tracks)
└── test
     └── ... (225 tracks)
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