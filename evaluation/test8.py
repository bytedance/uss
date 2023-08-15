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


def evaluate(args):
    
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

    if dataset_type in ["fsdkaggle2018", "fsd50k", "slakh2100"]:

        sub_dirs = sorted(list(Path(audios_dir).glob("*")))

        metrics_dict = {}

        for dir_index, sub_dir in enumerate(sub_dirs):

            print("------ {} ------".format(dir_index))

            label = re.search('=(.*)', sub_dir.name).group(1)

            query_emb_path = Path(query_embs_dir, Path(config_yaml).stem, "{}.pkl".format(label))
            query_emb = pickle.load(open(query_emb_path, 'rb'))

            mixture_paths = sorted(Path(audios_dir, sub_dir).glob("*mixture.wav"))

            metrics_dict[label] = {"sdr": []}

            for mixture_path in mixture_paths:

                mixture_path = str(mixture_path)
                source_path = mixture_path.replace('mixture.wav', 'source.wav')

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

                sdr = calculate_sdr(ref=source, est=sep_audio)

                print(source_path, sdr)

                metrics_dict[label]["sdr"].append(sdr)
                break

    elif dataset_type in ["voicebank-demand"]:

        label = "speech"

        query_emb_path = Path(query_embs_dir, Path(config_yaml).stem, "{}.pkl".format(label))
        query_emb = pickle.load(open(query_emb_path, 'rb'))

        source_paths = sorted(list(Path(audios_dir, "clean_testset_wav").glob("*.wav")))

        metrics_dict = {"speech": {"sdr": [], "pesq": [], "ssnr": [], "csig": [], "cbak": [], "covl": []}}

        for source_path in source_paths:

            # source_path = str(source_path)
            # mixture_path = source_path.replace
            mixture_path = Path(audios_dir, "noisy_testset_wav", Path(source_path).name)

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

            sdr = calculate_sdr(ref=source, est=sep_audio)


            EVALUATE_SR = 16000
            clean_16k = librosa.resample(y=source, orig_sr=sample_rate, target_sr=EVALUATE_SR)
            sep_wav_16k = librosa.resample(y=sep_audio, orig_sr=sample_rate, target_sr=EVALUATE_SR)

            pesq_ = pesq(EVALUATE_SR, clean_16k, sep_wav_16k, 'wb')
            (csig, cbak, covl) = pysepm.composite(clean_16k, sep_wav_16k, EVALUATE_SR)
            ssnr = pysepm.SNRseg(clean_16k, sep_wav_16k, EVALUATE_SR)

            metrics_dict[label]["sdr"].append(sdr)
            metrics_dict[label]["pesq"].append(pesq_)
            metrics_dict[label]["ssnr"].append(ssnr)
            metrics_dict[label]["csig"].append(csig)
            metrics_dict[label]["cbak"].append(cbak)
            metrics_dict[label]["covl"].append(covl)

            break

    elif dataset_type in ["musdb18"]:

        labels = ["vocals", "bass", "drums", "other"]

        metrics_dict = {}

        for label in labels:

            metrics_dict[label] = {"sdr": []}

            query_emb_path = Path(query_embs_dir, Path(config_yaml).stem, "{}.pkl".format(label))
            query_emb = pickle.load(open(query_emb_path, 'rb'))

            sub_dirs = sorted(list(Path(audios_dir).glob("*")))

            for sub_dir in sub_dirs:

                mixture_path = Path(sub_dir, "mixture.wav")

                # mixture_paths = [Path(sub_dir, "mixture.wav") for sub_dir in sub_dirs]
                # source_paths = [Path(sub_dir, "{}.wav".format(label)) for sub_dir in sub_dirs]

                # for mixture_path in mixture_paths:

                mixture_path = str(mixture_path)
                source_path = mixture_path.replace("mixture.wav", "{}.wav".format(label))

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

                t1 = time.time()
                (sdrs, _, _, _) = museval.evaluate(
                    references=source[None, :, None], 
                    estimates=sep_audio[None, :, None],
                    win=sample_rate,
                    hop=sample_rate,
                )
                print(time.time() - t1)

                metrics_dict[label]["sdr"].append(np.nanmedian(sdrs))

                from IPython import embed; embed(using=False); os._exit(0)

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

    # #
    # median_sdrs = []
    # for label in sdrs_dict.keys():
    #     median_sdr = np.nanmedian(sdrs_dict[label])
    #     print("{}: {:.4f}".format(label, median_sdr))
    #     median_sdrs.append(median_sdr)

    # print("--------")
    # print("Avg SDR: {:.4f}".format(np.nanmean(median_sdrs)))

            
    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--dataset_type', type=str)
    parser.add_argument('--audios_dir', type=str)
    parser.add_argument('--query_embs_dir', type=str)
    parser.add_argument('--device', type=str)

    args = parser.parse_args()
    
    evaluate(args)