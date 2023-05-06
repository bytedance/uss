import argparse
import os
import pathlib

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


def create_evaluation_meta(args):
    r"""Create csv containing information of anchor segments for creating
    mixtures. For each sound class k, we select M anchor segments that will be
    randomly mixed with anchor segments that do not contain sound class k. In
    total, there are classes_num x M mixtures to separate. Anchor segments are
    short segments (such as 2 s) detected by a pretrained sound event detection
    system on 10-second audio clips from AudioSet. All time stamps of anchor
    segments are written into a csv file. E.g.,

    .. code-block:: csv
        index_in_hdf5   audio_name  bgn_sample  end_sample  class_id    mix_rank
        4768    YC0j69NCIKfw.wav    140480  204480  347 0
        15640   Yip4ZCCgoVXc.wav    81920   145920  496 1
        10614   YTRxF5y6hFbE.wav    130240  194240  270 0
        9969    YRN1ho4G-W0o.wav    256000  320000  305 1
        ...

    When creating mixtures, for example:
        mixture_0 = YC0j69NCIKfw.wav + Yip4ZCCgoVXc.wav
        mixture_1 = YTRxF5y6hFbE.wav + YRN1ho4G-W0o.wav
        ...

    Args:
        workspace: str, path
        split: str, 'balanced_train' | 'test'
        gpus: int
        config_yaml: str, path of config file
    """

    # arguments & parameters
    workspace = args.workspace
    split = args.split
    output_audios_dir = args.output_audios_dir
    output_meta_csv_path = args.output_meta_csv_path

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
    steps_per_epoch = 10000
    num_workers = 16
    device = "cuda"
    sed_model_type = "Cnn14_DecisionLevelMax"

    if split == 'balanced_train':
        indexes_hdf5_path = os.path.join(
            workspace, "hdf5s/indexes/balanced_train.h5")

    elif split == 'test':
        indexes_hdf5_path = os.path.join(workspace, "hdf5s/indexes/eval.h5")
    # E.g., indexes_hdf5 looks like: {
    #     'audio_name': (audios_num,),
    #     'hdf5_path': (audios_num,),
    #     'index_in_hdf5': (audios_num,),
    #     'target': (audios_num, classes_num)
    # }

    sed_model = load_pretrained_panns(
        model_type=sed_model_type,
        checkpoint_path=get_path(panns_paths_dict[sed_model_type]),
        freeze=True,
    ).to(device)

    # dataset
    dataset = Dataset(
        steps_per_epoch=steps_per_epoch,
    )

    # sampler
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
        sub_dir = os.path.join(
            output_audios_dir,
            "class_id={}".format(class_id))
        os.makedirs(sub_dir, exist_ok=True)

    for batch_index, batch_data_dict in enumerate(dataloader):

        batch_data_dict['waveform'] = batch_data_dict['waveform'].to(device)
        # (batch_size, clip_samples)

        segments_dict = anchor_segment_detector(
            waveforms=batch_data_dict['waveform'],
            class_ids=batch_data_dict['class_id'],
        )

        mixtures, segments = anchor_segment_mixer(
            waveforms=segments_dict['waveform'],
        )

        mixtures = mixtures.data.cpu().numpy()
        segments = segments.data.cpu().numpy()

        source_names = batch_data_dict['audio_name']
        class_ids = segments_dict['class_id']
        bgn_samples = segments_dict['bgn_sample'].data.cpu().numpy()

        for n in range(batch_size):

            class_id = class_ids[n]

            if count_dict[class_id] < eval_segments_per_class:

                mixture_name = "class_id={},index={:03d},mixture.wav".format(
                    class_id, count_dict[class_id])
                source_name = "class_id={},index={:03d},source.wav".format(
                    class_id, count_dict[class_id])

                mixture_path = os.path.join(
                    output_audios_dir,
                    "class_id={}".format(class_id),
                    mixture_name)
                source_path = os.path.join(
                    output_audios_dir,
                    "class_id={}".format(class_id),
                    source_name)

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

                ###
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

    for class_id in count_dict.keys():
        if count_dict[class_id] < segments_per_class:
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

    os.makedirs(os.path.dirname(output_meta_csv_path), exist_ok=True)

    with open(output_meta_csv_path, 'w') as fw:

        fw.write(','.join(keys) + "\n")

        for n in range(items_num):

            fw.write(",".join([str(meta_dict[key][n]) for key in keys]) + "\n")

    print('Write out to {}'.format(output_meta_csv_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="mode")

    parser_train = subparsers.add_parser("create_evaluation_meta")
    parser_train.add_argument("--workspace", type=str, required=True)
    parser_train.add_argument(
        "--split",
        type=str,
        required=True,
        choices=[
            'balanced_train',
            'test'])
    parser_train.add_argument("--output_audios_dir", type=str, required=True)
    parser_train.add_argument(
        "--output_meta_csv_path",
        type=str,
        required=True)

    args = parser.parse_args()
    args.filename = pathlib.Path(__file__).stem

    if args.mode == "create_evaluation_meta":
        create_evaluation_meta(args)

    else:
        raise Exception("Error argument!")
