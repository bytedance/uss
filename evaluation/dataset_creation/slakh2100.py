import argparse
import time
from pathlib import Path
from typing import List, NoReturn, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile
import yaml
from evaluation.dataset_creation.audioset import (all_classes_finished,
                                                  write_meta_dict_to_csv)
from uss.config import SAMPLE_RATE


def parse_meta_csv(meta_csv: str) -> Tuple[List[str], List[str]]:
    r"""Parse csv file.

    Args:
        meta_csv: str, path of csv file

    Returns:
        audio_names: List[str]
        labels: List[str]
    """

    df = pd.read_csv(meta_csv, sep=',')
    audio_names = df["fname"].values
    labels = df["label"].values

    return audio_names, labels

    
def create_evaluation_data(args) -> NoReturn:
    r"""Create 2-second <mixture, source> pairs for evaluation.

    Args:
        dataset_dir: str, directory of the Slakh2100 dataset.
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

    eval_segments_per_class = 100
    random_state = np.random.RandomState(1234)

    if split in ["train", "test"]:
        sub_dirs = sorted(list(Path(dataset_dir, split).glob("*")))

    else:
        raise NotImplementedError

    total_time = time.time()

    # The candidate_dict contains <onset, offset> pairs for each plugin_name.
    candidates_dict = {}
    # E.g., {"funk_kit.nkm":
    #  [('datasets/slakh2100/train/Track00011/stem/S08.flac',
    #   'datasets/slakh2100/train/Track00011/mix.flac',
    #   132.0, 
    #   134.0), ...]
    #   ...}

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

            # Load source and detect active frames
            source, origin_sr = librosa.load(path=source_path, sr=None, mono=True)

            origin_segment_samples = int(origin_sr * segment_seconds)

            segments = librosa.util.frame(
                source, 
                frame_length=origin_segment_samples, 
                hop_length=origin_segment_samples
            )

            energy_array = np.max(np.abs(segments), axis=0)

            # Add active frames to candidates_dict
            bgn_end_pairs = []

            for i in range(len(energy_array)):
                if energy_array[i] > 0.1:
                    bgn_end_pairs.append((str(source_path), str(mixture_path), i * segment_seconds, (i + 1) * segment_seconds))


            if plugin_name not in candidates_dict.keys():
                candidates_dict[plugin_name] = bgn_end_pairs
            else:
                candidates_dict[plugin_name].extend(bgn_end_pairs)
    
    count_dict = {key: 0 for key in candidates_dict.keys()}

    meta_dict = {}
    meta_dict['source1_name'] = []
    meta_dict['source1_label'] = []
    meta_dict["source1_begin_second"] = []

    for label in candidates_dict.keys():

        if len(candidates_dict[label]) > 10:

            if len(candidates_dict[label]) < eval_segments_per_class:

                bgn_end_pairs = candidates_dict[label]

            else:
                indexes = random_state.choice(np.arange(len(candidates_dict[label])), size=eval_segments_per_class, replace=False)
                bgn_end_pairs = [candidates_dict[label][i] for i in indexes]

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

                meta_dict["source1_name"].append(Path(_mixture_path).parent.name)
                meta_dict["source1_label"].append(plugin_name)
                meta_dict["source1_begin_second"].append(begin_second)

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
