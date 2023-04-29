import argparse
import os
import time
import pickle
from pathlib import Path
from typing import Dict, List

import librosa
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
import torch.nn as nn

from casa.config import ID_TO_IX, LB_TO_IX, IX_TO_LB, csv_paths_dict, panns_paths_dict, model_paths_dict
from casa.models.pl_modules import LitSeparation, get_model_class
from casa.models.query_nets import initialize_query_net
from casa.parse_ontology import Node, get_ontology_tree
from casa.utils import (get_audioset632_id_to_lb, load_pretrained_panns,
                        parse_yaml, remove_silence, repeat_to_length, get_path)
from casa.inference import load_ss_model, calculate_query_emb


def add(args):

    dataset_root = "/mnt/bd/kqq3/datasets/musdb18hq/train"

    audio_names = sorted(os.listdir(dataset_root))

    # source_type = "drums"
    source_types = ["vocals", "bass", "drums", "other"]

    for source_type in source_types:

        new_dir = "./_tmp_wav/{}".format(source_type)
        Path(new_dir).mkdir(parents=True, exist_ok=True)

        for audio_name in audio_names:
            audio_path = os.path.join(dataset_root, audio_name, "{}.wav".format(source_type))
            new_path = Path(new_dir, "{}.wav".format(audio_name))

            string = 'ln -s "{}" "{}"'.format(audio_path, new_path)
            os.system(string)
            print(string)

    from IPython import embed; embed(using=False); os._exit(0)


def calcualte_condition(args) -> None:
    r"""Do separation for active sound classes."""

    # Arguments & parameters
    queries_dir = args.queries_dir
    query_emb_path = args.query_emb_path
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    # output_dir = args.output_dir

    device = "cuda"

    configs = parse_yaml(config_yaml)
    sample_rate = configs["data"]["sample_rate"]
    segment_seconds = configs["data"]["segment_seconds"]
    segment_samples = int(sample_rate * segment_seconds)

    # Load pretrained universal source separation model
    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
    ).to(device)

    query_condition = calculate_query_emb(
        queries_dir=queries_dir,
        pl_model=pl_model,
        sample_rate=sample_rate,
        remove_sil=True,
        segment_samples=segment_samples,
    )
    
    pickle_path = os.path.join("./query_conditions", "config={}".format(Path(config_yaml).stem), "{}.pkl".format(Path(queries_dir).stem))
    
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

    pickle.dump(query_condition, open(pickle_path, 'wb'))
    print("Write query condition to {}".format(pickle_path))

    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_dir", type=str, default="")
    parser.add_argument("--query_emb_path", type=str, default="")
    parser.add_argument("--config_yaml", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")

    args = parser.parse_args()

    # add(args)
    calcualte_condition(args)