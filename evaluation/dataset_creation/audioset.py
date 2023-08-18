import argparse
import multiprocessing
from typing import Dict, NoReturn
from pathlib import Path

import numpy as np
import soundfile
from torch.utils.data import DataLoader

from uss.config import (CLASSES_NUM, CLIP_SECONDS, FRAMES_PER_SECOND,
                        SAMPLE_RATE, panns_paths_dict)
from uss.data.anchor_segment_detectors import AnchorSegmentDetector
from uss.data.anchor_segment_mixers import AnchorSegmentMixer
from uss.data.datamodules import collate_fn
from uss.data.datasets import Dataset
from uss.data.samplers import BalancedSampler
from uss.utils import get_path, load_pretrained_panns


def create_evaluation_data(args) -> NoReturn:
    r"""Create 52,700 2-second <mixture, source> pairs from the balanced train / 
    test set of AudioSet for evaluation. Each sound class contains 100 pairs 
    for evaluation. The 2-second segment is mined by using a trained sound event 
    detection (SED) system.

    When creating mixtures:
        mixture = SED(YC0j69NCIKfw.wav) + scale(SED(Yip4ZCCgoVXc.wav))
        source = SED(YC0j69NCIKfw.wav)

    Args:
        indexes_hdf5_path: str, path
        output_audios_dir: str, directory to write out audios
        output_meta_csv_path: str, path to write out csv file
        device: str, e.g., "cuda" | "cpu"

    Returns:
        NoReturn
    """

    # Arguments & parameters
    indexes_hdf5_path = args.indexes_hdf5_path
    output_audios_dir = args.output_audios_dir
    output_meta_csv_path = args.output_meta_csv_path
    device = args.device

    sample_rate = SAMPLE_RATE
    frames_per_second = FRAMES_PER_SECOND
    clip_seconds = CLIP_SECONDS
    classes_num = CLASSES_NUM

    eval_segments_per_class = 100
    segment_seconds = 2.
    anchor_segment_detect_mode = "max_area"
    match_energy = True
    mix_num = 2

    batch_size = 32
    steps_per_epoch = 10000  # dummy value for data loader
    num_workers = min(16, multiprocessing.cpu_count())
    sed_model_type = "Cnn14_DecisionLevelMax"

    # Load sound event detection mdoel.
    sed_model = load_pretrained_panns(
        model_type=sed_model_type,
        checkpoint_path=get_path(panns_paths_dict[sed_model_type]),
        freeze=True,
    ).to(device)

    # Dataset
    dataset = Dataset(
        steps_per_epoch=steps_per_epoch,
    )

    # Sampler
    sampler = BalancedSampler(
        indexes_hdf5_path=indexes_hdf5_path,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    anchor_segment_detector = AnchorSegmentDetector(
        sed_model=sed_model,
        clip_seconds=clip_seconds,
        segment_seconds=segment_seconds,
        frames_per_second=frames_per_second,
        sample_rate=sample_rate,
        detect_mode=anchor_segment_detect_mode,
    ).to(device)

    anchor_segment_mixer = AnchorSegmentMixer(
        mix_num=mix_num,
        match_energy=match_energy,
    ).to(device)

    count_dict = {class_id: 0 for class_id in range(classes_num)}

    meta_dict = {
        'audio_name': [],
    }

    for i in range(mix_num):
        meta_dict['source{}_name'.format(i + 1)] = []
        meta_dict['source{}_class_id'.format(i + 1)] = []
        meta_dict['source{}_onset'.format(i + 1)] = []

    for class_id in range(classes_num):
        sub_dir = Path(output_audios_dir, "class_id={}".format(class_id))
        Path(sub_dir).mkdir(parents=True, exist_ok=True)

    for batch_index, batch_data_dict in enumerate(dataloader):

        batch_data_dict['waveform'] = batch_data_dict['waveform'].to(device)
        # (batch_size, clip_samples)

        segments_dict = anchor_segment_detector(
            waveforms=batch_data_dict['waveform'],
            class_ids=batch_data_dict['class_id'],
        )
        # {"waveform": (batch_size, segment_samples),
        #  "class_id": (batch_size,),
        #  "bgn_sample": (batch_size,),
        #  "end_sample": (batch_size,)
        # }

        mixtures, segments = anchor_segment_mixer(
            waveforms=segments_dict['waveform'],
        )
        # mixtures: (batch_size, segment_samples)
        # segments: (batch_size, segment_samples)

        mixtures = mixtures.data.cpu().numpy()
        segments = segments.data.cpu().numpy()

        source_names = batch_data_dict['audio_name']    # (batch_size,)
        class_ids = segments_dict['class_id']   # (batch_size)
        bgn_samples = segments_dict['bgn_sample'].data.cpu().numpy()    # (batch_size)

        for n in range(batch_size):

            class_id = class_ids[n]

            if count_dict[class_id] < eval_segments_per_class:

                # Paths to write out wavs
                mixture_name = "class_id={},index={:03d},mixture.wav".format(
                    class_id, count_dict[class_id])

                source_name = "class_id={},index={:03d},source.wav".format(
                    class_id, count_dict[class_id])

                mixture_path = Path(output_audios_dir, 
                    "class_id={}".format(class_id), mixture_name)

                source_path = Path(
                    output_audios_dir,
                    "class_id={}".format(class_id),
                    source_name)

                # Write out mixture and source
                soundfile.write(
                    file=mixture_path,
                    data=mixtures[n],
                    samplerate=sample_rate)

                soundfile.write(
                    file=source_path,
                    data=segments[n],
                    samplerate=sample_rate)

                print("Write out to {}".format(mixture_path))
                print("Write out to {}".format(source_path))

                # Write mixing information into a csv file.
                meta_dict['audio_name'].append(mixture_name)

                for i in range(mix_num):
                    meta_dict['source{}_name'.format(
                        i + 1)].append(source_names[(n + i) % batch_size])

                    meta_dict['source{}_onset'.format(
                        i + 1)].append(bgn_samples[(n + i) % batch_size] / sample_rate)

                    meta_dict['source{}_class_id'.format(
                        i + 1)].append(class_ids[(n + i) % batch_size])

                meta_dict['audio_name'].append(source_name)
                meta_dict['source1_name'].append(source_names[n])
                meta_dict['source1_onset'].append(bgn_samples[n] / sample_rate)
                meta_dict['source1_class_id'].append(class_ids[n])

                for i in range(1, mix_num):
                    meta_dict['source{}_name'.format(i + 1)].append("")
                    meta_dict['source{}_onset'.format(i + 1)].append("")
                    meta_dict['source{}_class_id'.format(i + 1)].append("")

                count_dict[class_id] += 1

        finished_n = np.sum([count_dict[class_id]
                            for class_id in range(classes_num)])

        print('Finished: {} / {}'.format(finished_n,
              eval_segments_per_class * classes_num))

        if all_classes_finished(count_dict, eval_segments_per_class):
            break

    write_meta_dict_to_csv(meta_dict, output_meta_csv_path)
    
    print("Write csv to {}".format(output_meta_csv_path))


def all_classes_finished(count_dict: Dict, segments_per_class: int) -> bool:
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

    for class_id in count_dict.keys():
        if count_dict[class_id] < segments_per_class:
            return False

    return True


def write_meta_dict_to_csv(meta_dict: Dict, output_meta_csv_path: str) -> NoReturn:
    r"""Write meta dict into a csv file.

    Args:
        meta_dict: dict, e.g., {
            "audio_name": (segments_num,),
            "source1_name": (segments_num,),
            "source1_class_id": (segments_num,),
            "source1_onset": (segments_num,),
            "source2_name": (segments_num,),
            "source2_class_id": (segments_num,),
            "source2_onset": (segments_num,),
        }
        output_csv_path: str, path to write out the csv file

    Returns:
        NoReturn
    """

    keys = list(meta_dict.keys())
    segments_num = len(meta_dict[keys[0]])

    Path(output_meta_csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_meta_csv_path, 'w') as fw:

        fw.write(','.join(keys) + "\n")

        for n in range(segments_num):

            fw.write(",".join([str(meta_dict[key][n]) for key in keys]) + "\n")
    
    print('Write out to {}'.format(output_meta_csv_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser = argparse.ArgumentParser()
    parser.add_argument("--indexes_hdf5_path", type=str, required=True)
    parser.add_argument("--output_audios_dir", type=str, required=True)
    parser.add_argument(
        "--output_meta_csv_path",
        type=str,
        required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda")

    args = parser.parse_args()
    
    create_evaluation_data(args)
