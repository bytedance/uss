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
from casa.utils import get_path

print(1234)

'''
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

    args.config_yaml = get_path(meta=model_paths_dict[condition_type]["config_yaml"])
    args.checkpoint_path = get_path(meta=model_paths_dict[condition_type]["checkpoint"])

    separate(args)


if __name__ == "__main__":
    print('asdf')
    add()
'''

'''
if __name__ == "__main__":
    
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

    args.config_yaml = get_path(meta=model_paths_dict[condition_type]["config_yaml"])
    args.checkpoint_path = get_path(meta=model_paths_dict[condition_type]["checkpoint"])

    separate(args)
'''