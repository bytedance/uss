import argparse
import os
import time
import pickle
import pathlib
from typing import Dict, List

import librosa
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
import torch.nn as nn
from pathlib import Path

from casa.inference import separate


model_configs = {
    "at_soft": {

        "config_yaml": Path(Path.home(), ".cache/casa/scripts/ss_model=resunet30,querynet=at_soft,data=full.yaml"),

        "config_yaml_size": 1659,

        "remote_config_yaml": "https://sandbox.zenodo.org/record/1186980/files/ss_model%3Dresunet30%2Cquerynet%3Dat_soft%2Cdata%3Dfull.yaml?download=1",

        "checkpoint_path": Path(Path.home(), ".cache/casa/checkpoints/ss_model=resunet30,querynet=at_soft,data=full,devices=8,step=100000.ckpt"),

        "checkpoint_size": 1121024828,

        "remote_checkpoint_path": "https://sandbox.zenodo.org/record/1186898/files/ss_model%3Dresunet30%2Cquerynet%3Dat_soft%2Cdata%3Dfull%2Cdevices%3D8%2Cstep%3D100000.ckpt?download=1",
    }
}


def download_models(meta_dict, re_download=False):

    config_yaml = meta_dict["config_yaml"]
    config_yaml_size = meta_dict["config_yaml_size"]
    remote_config_yaml = meta_dict["remote_config_yaml"]
    checkpoint_path = meta_dict["checkpoint_path"]
    checkpoint_size = meta_dict["checkpoint_size"]
    remote_checkpoint_path = meta_dict["remote_checkpoint_path"]

    if not Path(config_yaml).is_file() or Path(config_yaml).stat().st_size != config_yaml_size or re_download:
        Path(config_yaml).parents[0].mkdir(parents=True, exist_ok=True)
        os.system("wget -O {} {}".format(config_yaml, remote_config_yaml))
        print("Download to {}".format(config_yaml))

    if not Path(checkpoint_path).is_file() or Path(checkpoint_path).stat().st_size != checkpoint_size or re_download:
        Path(checkpoint_path).parents[0].mkdir(parents=True, exist_ok=True)
        os.system("wget -O {} {}".format(checkpoint_path, remote_checkpoint_path))
        print("Download to {}".format(checkpoint_path))


def add():

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str)
    parser.add_argument("--condition_type", type=str, default="at_soft")
    parser.add_argument("--levels", nargs="*", type=int, default=[])
    parser.add_argument("--class_ids", nargs="*", type=int, default=[])
    parser.add_argument("--queries_dir", type=str, default="")
    parser.add_argument("--query_emb_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")

    args = parser.parse_args()

    condition_type = args.condition_type
    meta_dict = model_configs[condition_type]

    args.config_yaml = meta_dict["config_yaml"]
    args.checkpoint_path = meta_dict["checkpoint_path"]

    download_models(meta_dict=meta_dict)

    separate(args)


# if __name__ == "__main__":
#     print('asdf')
#     add()