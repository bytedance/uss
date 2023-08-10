import argparse
import os
import time
import pickle
import csv
# import pathlib
from pathlib import Path

import h5py
import soundfile
import torch
import librosa
import numpy as np
import pandas as pd

from uss.data.anchor_segment_mixers import get_energy_ratio
from uss.utils import trunc_or_repeat_to_length
from uss.config import SAMPLE_RATE


def parse_meta_csv(meta_csv):

    df = pd.read_csv(meta_csv, sep=',')
    audio_names = df["fname"].values
    labels = df["label"].values

    return audio_names, labels

    
def create_evaluation_data(args):

    dataset_dir = args.dataset_dir
    split = args.split
    output_audios_dir = args.output_audios_dir
    output_meta_csv_path = args.output_meta_csv_path

    sample_rate = SAMPLE_RATE
    segment_seconds = 2.
    segment_samples = int(segment_seconds * sample_rate)

    mix_num = 2
    eval_segments_per_class = 100
    random_state = np.random.RandomState(1234)

    audios_dir = Path(dataset_dir, "FSDKaggle2018.audio_{}".format(split))

    if split == "test":
        meta_csv_path = Path(dataset_dir, "FSDKaggle2018.meta", "test_post_competition_scoring_clips.csv")
    else:
        raise NotImplementedError

    audio_names, labels = parse_meta_csv(meta_csv_path)
    audios_num = len(audio_names)

    count_dict = {label: 0 for label in set(labels)}

    meta_dict = {}

    for i in range(mix_num):
        meta_dict['source{}_name'.format(i + 1)] = []
        meta_dict['source{}_label'.format(i + 1)] = []

    meta_dict["source2_scale_factor"] = []

    total_time = time.time()

    while True:

        indexes = random_state.permutation(audios_num - 1)

        for i in indexes:

            if labels[i] != labels[i + 1]:

                label = labels[i]

                if count_dict[label] < eval_segments_per_class:

                    source1_path = Path(audios_dir, audio_names[i])
                    source2_path = Path(audios_dir, audio_names[i + 1])

                    source1, _ = librosa.core.load(source1_path, sr=sample_rate, mono=True)
                    source2, _ = librosa.core.load(source2_path, sr=sample_rate, mono=True)

                    segment1 = trunc_or_repeat_to_length(
                        audio=source1, 
                        segment_samples=segment_samples
                    )

                    segment2 = trunc_or_repeat_to_length(
                        audio=source2, 
                        segment_samples=segment_samples
                    )
                    
                    ratio = get_energy_ratio(
                        segment1=torch.Tensor(segment1), 
                        segment2=torch.Tensor(segment2),
                    ).item()
                    
                    segment2 *= ratio

                    mixture = segment1 + segment2

                    # soundfile.write(file="_zz1.wav", data=source1, samplerate=sample_rate)
                    # soundfile.write(file="_zz2.wav", data=source2, samplerate=sample_rate)

                    mixture_name = "label={},index={:03d},mixture.wav".format(
                        label, count_dict[label])

                    source_name = "label={},index={:03d},source.wav".format(
                        label, count_dict[label])

                    mixture_path = Path(
                        output_audios_dir,
                        "label={}".format(label),
                        mixture_name)

                    source_path = Path(
                        output_audios_dir,
                        "label={}".format(label),
                        source_name)

                    Path(mixture_path).parent.mkdir(parents=True, exist_ok=True)

                    soundfile.write(
                        file=mixture_path,
                        data=mixture,
                        samplerate=sample_rate)
                    soundfile.write(
                        file=source_path,
                        data=segment1,
                        samplerate=sample_rate)

                    print("Write out to {}".format(mixture_path))
                    print("Write out to {}".format(source_path))
                    print("{}: {} / {}".format(label, count_dict[label], eval_segments_per_class))

                    meta_dict["source1_name"].append(audio_names[i])
                    meta_dict["source2_name"].append(audio_names[i + 1])

                    meta_dict["source1_label"].append(labels[i])
                    meta_dict["source2_label"].append(labels[i + 1])

                    meta_dict["source2_scale_factor"].append(ratio)

                    count_dict[label] += 1

        if all_classes_finished(count_dict, eval_segments_per_class):
            break

    write_meta_dict_to_csv(meta_dict, output_meta_csv_path)
    print("Write csv to {}".format(output_meta_csv_path))

    print('Time: {:.3f} s'.format(time.time() - total_time))


def all_classes_finished(count_dict, segments_per_class):
    r"""Check if all sound classes have #segments_per_class segments in
    count_dict.

    Args:
        count_dict: dict, e.g., {
            0: 12,
            1: 4,
            ...,
            526: 33,
        }
        segments_per_class: int

    Returns:
        bool
    """

    for label in count_dict.keys():
        if count_dict[label] < segments_per_class:
            return False

    return True


def write_meta_dict_to_csv(meta_dict, output_meta_csv_path):
    r"""Write meta dict into a csv file.

    Args:
        meta_dict: dict, e.g., {
            'index_in_hdf5': (segments_num,),
            'audio_name': (segments_num,),
            'class_id': (segments_num,),
        }
        output_csv_path: str
    """

    keys = list(meta_dict.keys())

    items_num = len(meta_dict[keys[0]])

    Path(output_meta_csv_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_meta_csv_path, 'w') as fw:

        fw.write(','.join(keys) + "\n")

        for n in range(items_num):

            fw.write(",".join([str(meta_dict[key][n]) for key in keys]) + "\n")

    print('Write out to {}'.format(output_meta_csv_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument("--output_audios_dir", type=str, required=True)
    parser.add_argument("--output_meta_csv_path", type=str, required=True)

    # Parse arguments.
    args = parser.parse_args()

    create_evaluation_data(args)
