import datetime
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import torch
import torch.nn as nn
import yaml
from panns_inference.models import Cnn14, Cnn14_DecisionLevelMax


def create_logging(log_dir, filemode):
    os.makedirs(log_dir, exist_ok=True)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, "{:04d}.log".format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, "{:04d}.log".format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        filename=log_path,
        filemode=filemode,
    )

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    return logging


def float32_to_int16(x: float) -> int:
    x = np.clip(x, a_min=-1, a_max=1)
    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x: int) -> float:
    return (x / 32767.0).astype(np.float32)


def parse_yaml(config_yaml: str) -> Dict:
    r"""Parse yaml file.

    Args:
        config_yaml (str): config yaml path

    Returns:
        yaml_dict (Dict): parsed yaml file
    """

    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


def get_audioset632_id_to_lb(ontology_path: str) -> Dict:
    r"""Get AudioSet 632 classes ID to label mapping."""

    audioset632_id_to_lb = {}

    with open(ontology_path) as f:
        data_list = json.load(f)

    for e in data_list:
        audioset632_id_to_lb[e["id"]] = e["name"]

    return audioset632_id_to_lb


def load_pretrained_panns(
    model_type: str,
    checkpoint_path: str,
    freeze: bool
) -> nn.Module:
    r"""Load pretrained pretrained audio neural networks (PANNs).

    Args:
        model_type: str, e.g., "Cnn14"
        checkpoint_path, str, e.g., "Cnn14_mAP=0.431.pth"
        freeze: bool

    Returns:
        model: nn.Module
    """

    if model_type == "Cnn14":
        Model = Cnn14

    elif model_type == "Cnn14_DecisionLevelMax":
        Model = Cnn14_DecisionLevelMax

    else:
        raise NotImplementedError

    model = Model(sample_rate=32000, window_size=1024, hop_size=320,
                  mel_bins=64, fmin=50, fmax=14000, classes_num=527)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model


def energy(x):
    return torch.mean(x ** 2)


def magnitude_to_db(x):
    eps = 1e-10
    return 20. * np.log10(max(x, eps))


def db_to_magnitude(x):
    return 10. ** (x / 20)


def ids_to_hots(ids, classes_num, device):
    hots = torch.zeros(classes_num).to(device)
    for id in ids:
        hots[id] = 1
    return hots


def calculate_sdr(
    ref: np.ndarray,
    est: np.ndarray,
    eps=1e-10
) -> float:
    r"""Calculate SDR between reference and estimation.

    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """

    noise = est - ref

    numerator = np.clip(a=np.mean(ref ** 2), a_min=eps, a_max=None)

    denominator = np.clip(a=np.mean(noise ** 2), a_min=eps, a_max=None)

    sdr = 10. * np.log10(numerator / denominator)

    return sdr


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path

        self.backup_statistics_path = "{}_{}.pkl".format(
            os.path.splitext(self.statistics_path)[0],
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )

        self.statistics_dict = {"balanced_train": [], "test": []}

    def append(self, steps, statistics, split, flush=True):
        statistics["steps"] = steps
        self.statistics_dict[split].append(statistics)

        if flush:
            self.flush()

    def flush(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, "wb"))
        pickle.dump(
            self.statistics_dict, open(
                self.backup_statistics_path, "wb"))
        logging.info("    Dump statistics to {}".format(self.statistics_path))
        logging.info(
            "    Dump statistics to {}".format(
                self.backup_statistics_path))


def get_mean_sdr_from_dict(sdris_dict):
    mean_sdr = np.nanmean(list(sdris_dict.values()))
    return mean_sdr


def remove_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    r"""Remove silent frames."""
    window_size = int(sample_rate * 0.1)
    threshold = 0.02

    frames = librosa.util.frame(
        x=audio,
        frame_length=window_size,
        hop_length=window_size).T
    # shape: (frames_num, window_size)

    new_frames = get_active_frames(frames, threshold)
    # shape: (new_frames_num, window_size)

    new_audio = new_frames.flatten()
    # shape: (new_audio_samples,)

    return new_audio


def get_active_frames(frames: np.ndarray, threshold: float) -> np.ndarray:
    r"""Get active frames."""

    energy = np.max(np.abs(frames), axis=-1)
    # shape: (frames_num,)

    active_indexes = np.where(energy > threshold)[0]
    # shape: (new_frames_num,)

    new_frames = frames[active_indexes]
    # shape: (new_frames_num,)

    return new_frames


def repeat_to_length(audio: np.ndarray, segment_samples: int) -> np.ndarray:
    r"""Repeat audio to length."""

    repeats_num = (segment_samples // audio.shape[-1]) + 1
    audio = np.tile(audio, repeats_num)[0: segment_samples]

    return audio


def get_path(meta, re_download=False):

    path = meta["path"]
    remote_path = meta["remote_path"]
    size = meta["size"]

    if not Path(path).is_file() or Path(
            path).stat().st_size != size or re_download:

        Path(path).parents[0].mkdir(parents=True, exist_ok=True)
        os.system("wget -O {} {}".format(path, remote_path))
        print("Download to {}".format(path))

    return path
