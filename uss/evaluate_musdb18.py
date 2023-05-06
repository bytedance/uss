import argparse
import os
import time
import pickle
from pathlib import Path
from typing import Dict, List

import museval
import librosa
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
import torch.nn as nn

from uss.config import ID_TO_IX, LB_TO_IX, IX_TO_LB, csv_paths_dict, panns_paths_dict
from uss.models.pl_modules import LitSeparation, get_model_class
from uss.models.query_nets import initialize_query_net
from uss.parse_ontology import Node, get_ontology_tree
from uss.utils import (get_audioset632_id_to_lb, load_pretrained_panns,
                        parse_yaml, remove_silence, repeat_to_length, get_path)
from uss.inference import load_ss_model, calculate_query_emb, separate_by_query_condition


def add(args):

    dataset_root = "/mnt/bd/kqq3/datasets/musdb18hq/train"

    audio_names = sorted(os.listdir(dataset_root))

    # source_type = "drums"
    source_types = ["vocals", "bass", "drums", "other", "mixture"]

    for source_type in source_types:

        new_dir = "./_tmp_wav/{}".format(source_type)
        Path(new_dir).mkdir(parents=True, exist_ok=True)

        for audio_name in audio_names:
            audio_path = os.path.join(dataset_root, audio_name, "{}.wav".format(source_type))
            new_path = Path(new_dir, "{}.wav".format(audio_name))

            string = 'ln -s "{}" "{}"'.format(audio_path, new_path)
            os.system(string)
            print(string)


def calcualte_condition(args) -> None:
    r"""Do separation for active sound classes."""

    # Arguments & parameters
    queries_dir = args.queries_dir
    query_emb_path = args.query_emb_path
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path

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
    
    pickle_path = Path("./query_conditions", "config={}".format(Path(config_yaml).stem), "{}.pkl".format(Path(queries_dir).stem))
    
    pickle_path.parent.mkdir(parents=True, exist_ok=True)

    pickle.dump(query_condition, open(pickle_path, 'wb'))
    print("Write query condition to {}".format(pickle_path))


def evaluate(args):
    r"""Do separation for active sound classes."""

    # Arguments & parameters
    query_emb_path = args.query_emb_path
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    # output_dir = args.output_dir

    # non_sil_threshold = 1e-6
    device = "cuda"
    # ontology_path = get_path(csv_paths_dict["ontology.csv"])

    configs = parse_yaml(config_yaml)
    sample_rate = configs["data"]["sample_rate"]
    segment_seconds = configs["data"]["segment_seconds"]
    segment_samples = int(sample_rate * segment_seconds)

    # Create directory
    # if not output_dir:
    #     output_dir = os.path.join(
    #         "separated_results",
    #         Path(audio_path).stem)

    # Load pretrained universal source separation model
    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
    ).to(device)

    # Load audio
    # audio, fs = librosa.load(path=audio_path, sr=sample_rate, mono=True)

    # Load pretrained audio tagging model
    # at_model_type = "Cnn14"
        
    query_condition = pickle.load(open(query_emb_path, 'rb'))

    audio_paths = sorted(list(Path("/home/tiger/datasets/musdb18hq/test").rglob("*/mixture.wav")))
    # target_paths = sorted(list(Path("/home/tiger/datasets/musdb18hq/test").rglob("*/{}.wav".format(Path(query_emb_path).stem))))

    # mixtures_dir = "./_tmp_wav/mixture"
    # targets_dir = Path("./_tmp_wav", Path(query_emb_path).stem)

    # from IPython import embed; embed(using=False); os._exit(0)

    # audio_paths = sorted(list(Path(mixtures_dir).glob("*.wav")))

    all_median_sdrs = []

    for n, audio_path in enumerate(audio_paths):

        audio, _ = librosa.load(path=audio_path, sr=sample_rate, mono=True)

        # target_path = Path(targets_dir, Path(audio_path.name))
        target_path = Path(Path(audio_path).parent, "{}.wav".format(Path(query_emb_path).stem))
        target, _ = librosa.load(path=target_path, sr=sample_rate, mono=True)

        output_path = Path("_tmp_sep", Path(config_yaml).stem, Path(query_emb_path).stem, "{}.wav".format(Path(audio_path).parent.name))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sep_audio = separate_by_query_condition(
            audio=audio, 
            segment_samples=segment_samples, 
            sample_rate=sample_rate,
            query_condition=query_condition,
            pl_model=pl_model,
            output_path=output_path,
        )

        sdrs, _, _, _ = museval.evaluate(target[None, :, None], sep_audio[None, :, None], win=sample_rate, hop=sample_rate)  # (nsrc, nsampl, nchan)

        print(np.nanmedian(sdrs))
        all_median_sdrs.append(np.nanmedian(sdrs))

    print("==========")
    print(np.median(all_median_sdrs))
    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_link = subparsers.add_parser("link")

    parser_a = subparsers.add_parser("calculate_condition")
    parser_a.add_argument("--queries_dir", type=str, default="")
    parser_a.add_argument("--query_emb_path", type=str, default="")
    parser_a.add_argument("--config_yaml", type=str, default="")
    parser_a.add_argument("--checkpoint_path", type=str, default="")

    parser_evaluate = subparsers.add_parser("evaluate")
    parser_evaluate.add_argument("--query_emb_path", type=str, default="")
    parser_evaluate.add_argument("--config_yaml", type=str, default="")
    parser_evaluate.add_argument("--checkpoint_path", type=str, default="")

    args = parser.parse_args()

    if args.mode == "link":
        add(args)

    elif args.mode == "calculate_condition":
        calcualte_condition(args)

    elif args.mode == "evaluate":
        evaluate(args)

    else:
        raise NotImplementedError