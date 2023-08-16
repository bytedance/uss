import argparse
import os
import time
import pickle
import csv
# import pathlib
import yaml
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

    eval_segments_per_class = 100
    random_state = np.random.RandomState(1234)

    if split in ["train", "test"]:
        sub_dirs = sorted(list(Path(dataset_dir, split).glob("*")))

    else:
        raise NotImplementedError

    total_time = time.time()

    tmp_dict = {}

    for n, sub_dir in enumerate(sub_dirs):
        print(sub_dir)

        # Parse meta yaml
        meta_yaml_path = Path(sub_dir, "metadata.yaml")

        with open(meta_yaml_path, 'r') as f:
            meta_yaml = yaml.load(f, Loader=yaml.FullLoader)

        stems_dict = meta_yaml["stems"]

        for key in stems_dict.keys():

            plugin_name = stems_dict[key]["plugin_name"]

            source_path = Path(sub_dir, "stems", "{}.flac".format(key))
            mixture_path = Path(sub_dir, "mix.flac")

            if not Path(source_path).is_file():
                continue

            source, origin_sr = librosa.load(path=source_path, sr=None, mono=True)

            origin_segment_samples = int(origin_sr * segment_seconds)

            segments = librosa.util.frame(
                source, 
                frame_length=origin_segment_samples, 
                hop_length=origin_segment_samples
            )

            energy_array = np.max(np.abs(segments), axis=0)

            sorted_indexes = np.argsort(energy_array)[::-1]

            candidate_indexes = []
            bgn_end_pairs = []

            '''
            for i in range(10):
                index = sorted_indexes[i]
                if energy_array[index] > 0.1:
                    candidate_indexes.append(index)
                    bgn_end_pairs.append((index * segment_seconds, (index + 1) * segment_seconds))
            '''
            for i in range(len(energy_array)):
                if energy_array[i] > 0.1:
                    bgn_end_pairs.append((str(source_path), str(mixture_path), i * segment_seconds, (i + 1) * segment_seconds))


            if plugin_name not in tmp_dict.keys():
                tmp_dict[plugin_name] = bgn_end_pairs
            else:
                tmp_dict[plugin_name].extend(bgn_end_pairs)

        # if n == 10:
        #     break

    count_dict = {key: 0 for key in tmp_dict.keys()}

    meta_dict = {}
    meta_dict['source1_name'] = []
    meta_dict['source1_label'] = []
    meta_dict["source1_begin_second"] = []

    for label in tmp_dict.keys():

        if len(tmp_dict[label]) > 10:

            if len(tmp_dict[label]) < eval_segments_per_class:

                bgn_end_pairs = tmp_dict[label]

            else:
                indexes = random_state.choice(np.arange(len(tmp_dict[label])), size=eval_segments_per_class, replace=False)
                bgn_end_pairs = [tmp_dict[label][i] for i in indexes]

            for i in range(len(bgn_end_pairs)):

                _source_path, _mixture_path, begin_second, end_second = bgn_end_pairs[i]

                source, origin_sr = librosa.load(path=_source_path, sr=None, mono=True)
                mixture, _ = librosa.load(path=_mixture_path, sr=None, mono=True)

                begin_sample = int(origin_sr * begin_second)
                end_sample = int(origin_sr * end_second)

                segment1 = source[begin_sample : end_sample]
                mixture = mixture[begin_sample : end_sample]

                segment1 = librosa.resample(segment1, orig_sr=origin_sr, target_sr=sample_rate)
                mixture = librosa.resample(mixture, orig_sr=origin_sr, target_sr=sample_rate)

                #
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

                meta_dict["source1_name"].append(Path(_mixture_path).parent.name)
                meta_dict["source1_label"].append(plugin_name)
                meta_dict["source1_begin_second"].append(begin_second)

                count_dict[label] += 1

        if all_classes_finished(count_dict, eval_segments_per_class):
            break

        # break 

    write_meta_dict_to_csv(meta_dict, output_meta_csv_path)
    # print("Write csv to {}".format(output_meta_csv_path))

    print('Time: {:.3f} s'.format(time.time() - total_time))
    from IPython import embed; embed(using=False); os._exit(0)


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
