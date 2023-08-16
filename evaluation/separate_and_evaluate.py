import re
import os
import pickle
import argparse
from pathlib import Path
import librosa
import numpy as np
from pesq import pesq
import pysepm
import museval
import time

from uss.inference import calculate_query_emb, load_ss_model, separate_by_query_condition
from uss.config import SAMPLE_RATE
from uss.utils import parse_yaml, calculate_sdr


def separate_and_evaluate(args):
    
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    dataset_type = args.dataset_type
    audios_dir = args.audios_dir
    query_embs_dir = args.query_embs_dir
    device = args.device

    sample_rate = SAMPLE_RATE
    segment_seconds = 2.
    segment_samples = int(sample_rate * segment_seconds)

    configs = parse_yaml(config_yaml)

    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
    ).to(device)

    if dataset_type in ["audioset", "fsdkaggle2018", "fsd50k", "slakh2100"]:

        paths_dict = get_2s_segments_paths_dict(
            audios_dir=audios_dir, 
            query_embs_dir=query_embs_dir, 
            config_yaml=config_yaml,
        )
        metric_types = ["sdr", "sdri"]

    elif dataset_type in ["musdb18"]:

        paths_dict = get_musdb18_paths_dict(
            audios_dir=audios_dir, 
            query_embs_dir=query_embs_dir, 
            config_yaml=config_yaml,
        )
        metric_types = ["musdb18_sdr", "musdb18_sdri"]

    elif dataset_type in ["voicebank-demand"]:

        paths_dict = get_voicebank_demand_paths_dict(
            audios_dir=audios_dir, 
            query_embs_dir=query_embs_dir, 
            config_yaml=config_yaml,
        )
        metric_types = ["pesq", "ssnr", "csig", "cbak", "covl"]

    else:
        raise NotImplementedError

    metrics_dict = add(
        paths_dict=paths_dict, 
        pl_model=pl_model, 
        segment_samples=segment_samples, 
        sample_rate=sample_rate, 
        metric_types=metric_types
    )

    # Print
    labels = metrics_dict.keys()
    median_metrics_dict = {}

    for label in labels:

        median_metrics_dict[label] = {}

        metric_types = metrics_dict[label].keys()

        string = label

        for metric_type in metric_types:
            median_metrics_dict[label][metric_type] = np.nanmedian(metrics_dict[label][metric_type])
            string += ", {}: {:.4f}".format(metric_type, median_metrics_dict[label][metric_type])
            
        print(string)

    #
    print("-------")
    string = "Avg"
    for metric_type in metric_types:
        avg_metric = np.nanmean([median_metrics_dict[label][metric_type] for label in labels])
        string += ", {}: {:.4f}".format(metric_type, avg_metric)
    print(string)


def get_2s_segments_paths_dict(audios_dir, query_embs_dir, config_yaml):

    sub_dirs = sorted(list(Path(audios_dir).glob("*")))

    paths_dict = {}

    for sub_dir in sub_dirs:

        label = re.search('=(.*)', sub_dir.name).group(1)
        query_emb_path = Path(query_embs_dir, "{}.pkl".format(label))

        paths_dict[label] = {
            "source_path": [],
            "mixture_path": [],
            "query_embedding_path": query_emb_path,
        }
    
        mixture_paths = sorted(Path(sub_dir).glob("*mixture.wav"))

        for mixture_path in mixture_paths:

            mixture_path = str(mixture_path)
            source_path = mixture_path.replace('mixture.wav', 'source.wav')

            paths_dict[label]["source_path"].append(source_path)
            paths_dict[label]["mixture_path"].append(mixture_path)

    return paths_dict


def get_musdb18_paths_dict(audios_dir, query_embs_dir, config_yaml):

    labels = ["vocals", "bass", "drums", "other"]
    paths_dict = {}

    for label in labels:

        query_emb_path = Path(query_embs_dir, "{}.pkl".format(label))

        paths_dict[label] = {
            "source_path": [],
            "mixture_path": [],
            "query_embedding_path": query_emb_path,
        }

        sub_dirs = sorted(list(Path(audios_dir).glob("*")))

        for sub_dir in sub_dirs:

            mixture_path = Path(sub_dir, "mixture.wav")
            mixture_path = str(mixture_path)
            source_path = mixture_path.replace("mixture.wav", "{}.wav".format(label))

            paths_dict[label]["source_path"].append(source_path)
            paths_dict[label]["mixture_path"].append(mixture_path)

    return paths_dict


def get_voicebank_demand_paths_dict(audios_dir, query_embs_dir, config_yaml):

    labels = ["speech"]
    paths_dict = {}

    for label in labels:

        query_emb_path = Path(query_embs_dir, "{}.pkl".format(label))

        paths_dict[label] = {
            "source_path": [],
            "mixture_path": [],
            "query_embedding_path": query_emb_path,
        }

        source_paths = sorted(list(Path(audios_dir, "clean_testset_wav").glob("*.wav")))

        for source_path in source_paths:

            mixture_path = Path(audios_dir, "noisy_testset_wav", Path(source_path).name)

            paths_dict[label]["source_path"].append(source_path)
            paths_dict[label]["mixture_path"].append(mixture_path)

    return paths_dict

def add(paths_dict, pl_model, segment_samples, sample_rate, metric_types):

    metrics_dict = {}

    for label in paths_dict.keys():

        query_emb_path = paths_dict[label]["query_embedding_path"]
        source_paths = paths_dict[label]["source_path"]
        mixture_paths = paths_dict[label]["mixture_path"]

        query_emb = pickle.load(open(query_emb_path, 'rb'))

        metrics_dict[label] = {metric_type: [] for metric_type in metric_types}

        audios_num = len(source_paths)

        for n in range(audios_num):

            source_path = source_paths[n]
            mixture_path = mixture_paths[n]

            mixture, _ = librosa.load(path=mixture_path, sr=sample_rate, mono=True)
            source, _ = librosa.load(path=source_path, sr=sample_rate, mono=True)

            sep_audio = separate_by_query_condition(
                audio=mixture,
                segment_samples=segment_samples,
                sample_rate=sample_rate,
                query_condition=query_emb,
                pl_model=pl_model,
                output_path=None,
            )

            metrics = calculate_metrics(
                ref=source, 
                est=sep_audio, 
                mix=mixture,
                metric_types=metric_types,
                sample_rate=sample_rate,
            )

            for metric_type in metric_types:
                metrics_dict[label][metric_type].append(metrics[metric_type])

            break
            
    return metrics_dict



def calculate_metrics(ref, est, mix, metric_types, sample_rate):

    metrics = {}

    for metric_type in metric_types:

        if metric_type == "sdr":
            sdr = calculate_sdr(ref=ref, est=est)
            metrics["sdr"] = sdr

        elif metric_type == "sdri":

            assert "sdr" in metrics.keys()
            sdr0 = calculate_sdr(ref=ref, est=mix)
            metrics["sdri"] = metrics["sdr"] - sdr0

        elif metric_type == "musdb18_sdr":

            (sdrs, _, _, _) = museval.evaluate(
                references=est[None, :, None], 
                estimates=ref[None, :, None],
                win=sample_rate,
                hop=sample_rate,
            )
            sdr = np.nanmedian(sdrs)
            metrics["musdb18_sdr"] = sdr

        elif metric_type == "musdb18_sdri":

            assert "musdb18_sdr" in metrics.keys()

            (sdrs0, _, _, _) = museval.evaluate(
                references=est[None, :, None], 
                estimates=ref[None, :, None],
                win=sample_rate,
                hop=sample_rate,
            )
            sdr0 = np.nanmedian(sdrs0)
            metrics["musdb18_sdri"] = metrics["musdb18_sdr"] - sdr0

        elif metric_type in ["pesq", "ssnr", "csig", "cbak", "covl"]:

            if metric_type in metrics.keys():
                continue

            evaluate_sr = 16000
            ref_16k = librosa.resample(y=ref, orig_sr=sample_rate, target_sr=evaluate_sr)
            est_16k = librosa.resample(y=est, orig_sr=sample_rate, target_sr=evaluate_sr)

            pesq_ = pesq(evaluate_sr, ref_16k, est_16k, 'wb')
            (csig, cbak, covl) = pysepm.composite(ref_16k, est_16k, evaluate_sr)
            ssnr = pysepm.SNRseg(ref_16k, est_16k, evaluate_sr)

            metrics["pesq"] = pesq_
            metrics["ssnr"] = ssnr
            metrics["csig"] = csig
            metrics["cbak"] = cbak
            metrics["covl"] = covl

        else:
            raise NotImplementedError

    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--dataset_type', type=str)
    parser.add_argument('--audios_dir', type=str)
    parser.add_argument('--query_embs_dir', type=str)
    parser.add_argument('--device', type=str)

    args = parser.parse_args()
    
    separate_and_evaluate(args)