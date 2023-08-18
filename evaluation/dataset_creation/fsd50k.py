import argparse
import time
from pathlib import Path
from typing import List, NoReturn, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile
import torch
from evaluation.dataset_creation.audioset import (all_classes_finished,
                                                  write_meta_dict_to_csv)
from uss.config import SAMPLE_RATE
from uss.data.anchor_segment_mixers import get_energy_ratio
from uss.utils import trunc_or_repeat_to_length


def parse_meta_csv(meta_csv: str) -> Tuple[List[str], List[str]]:
    r"""Parse csv file.

    Args:
        meta_csv: str, path of csv file

    Returns:
        audio_names: List[str]
        labels: List[str]
    """
    df = pd.read_csv(meta_csv, sep=',')
    audio_names = ["{}.wav".format(name) for name in df["fname"].values]
    labels = [label.split(",")[0] for label in df["labels"].values]
    
    return audio_names, labels

    
def create_evaluation_data(args) -> NoReturn:
    r"""Create 2-second <mixture, source> pairs for evaluation.

    Args:
        dataset_dir: str, directory of the FSD50k dataset.
        split: str, "train" | "test"
        output_audios_dir: str, directory to write out audios
        output_meta_csv_path: str, path to write out csv file

    Returns:
        NoReturn
    """

    # Args & parameters
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

    if split == "train":
        audios_dir = Path(dataset_dir, "FSD50K.dev_audio")
        meta_csv_path = Path(dataset_dir, "FSD50K.ground_truth", "dev.csv")

    elif split == "test":
        audios_dir = Path(dataset_dir, "FSD50K.eval_audio")
        meta_csv_path = Path(dataset_dir, "FSD50K.ground_truth", "eval.csv")

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

        indexes = random_state.permutation(audios_num)

        for i in indexes:

            i1 = i
            i2 = (i + 1) % audios_num

            if labels[i1] != labels[i2]:

                label = labels[i1]

                if count_dict[label] < eval_segments_per_class:

                    source1_path = Path(audios_dir, audio_names[i1])
                    source2_path = Path(audios_dir, audio_names[i2])

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

                    # Paths to write out wavs
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

                    # Write out mixture and source
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

                    meta_dict["source1_name"].append(audio_names[i1])
                    meta_dict["source2_name"].append(audio_names[i2])

                    meta_dict["source1_label"].append(labels[i1])
                    meta_dict["source2_label"].append(labels[i2])

                    meta_dict["source2_scale_factor"].append(ratio)

                    count_dict[label] += 1

        if all_classes_finished(count_dict, eval_segments_per_class):
            break

    write_meta_dict_to_csv(meta_dict, output_meta_csv_path)

    print('Time: {:.3f} s'.format(time.time() - total_time))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument("--output_audios_dir", type=str, required=True)
    parser.add_argument("--output_meta_csv_path", type=str, required=True)

    args = parser.parse_args()

    create_evaluation_data(args)
