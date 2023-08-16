import re
import os
import pickle
import argparse
from pathlib import Path
import museval

from uss.inference import calculate_query_emb, load_ss_model
from uss.config import SAMPLE_RATE
from uss.utils import parse_yaml


def calculate_embeddings(args):
    
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    dataset_type = args.dataset_type
    audios_dir = args.audios_dir
    output_embs_dir = args.output_embs_dir
    device = args.device

    sample_rate = SAMPLE_RATE
    segment_seconds = 2.
    segment_samples = int(sample_rate * segment_seconds)

    configs = parse_yaml(config_yaml)

    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
    ).to(device)

    paths_dict, remove_sil = get_paths_dict(dataset_type=dataset_type, audios_dir=audios_dir)

    labels = sorted(paths_dict.keys())

    for label in labels:

        query_paths = paths_dict[label]

        avg_query_condition = calculate_query_emb(
            queries_dir=None,
            pl_model=pl_model,
            sample_rate=sample_rate,
            remove_sil=remove_sil,
            segment_samples=segment_samples,
            batch_size=8,
            query_paths=query_paths,
        )

        output_emb_path = Path(output_embs_dir, "{}.pkl".format(label))
        Path(output_emb_path).parent.mkdir(parents=True, exist_ok=True)

        pickle.dump(avg_query_condition, open(output_emb_path, "wb"))
        print("Write out to {}".format(output_emb_path))


def get_paths_dict(dataset_type, audios_dir):

    if dataset_type in ["audioset", "fsdkaggle2018", "fsd50k", "slakh2100"]:

        sub_dirs = sorted(os.listdir(audios_dir))

        paths_dict = {}

        for sub_dir in sub_dirs:
            
            label = re.search('=(.*)', sub_dir).group(1)
            audio_paths = sorted(Path(audios_dir, sub_dir).glob("*source.wav"))
            
            paths_dict[label] = audio_paths

        remove_sil = False

    elif dataset_type in ["voicebank-demand"]:

        audio_paths = sorted(list(Path(audios_dir, "clean_trainset_wav").glob("*.wav")))

        paths_dict = {}
        label = "speech"
        paths_dict[label] = audio_paths

        remove_sil = True

    elif dataset_type in ["musdb18"]:

        labels = ["vocals", "bass", "drums", "other"]

        paths_dict = {}

        for label in labels:

            sub_dirs = sorted(list(Path(audios_dir).glob("*")))

            audio_paths = [Path(sub_dir, "{}.wav".format(label)) for sub_dir in sub_dirs]

            paths_dict[label] = audio_paths

        remove_sil = True

    else:
        raise NotImplementedError

    return paths_dict, remove_sil


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--dataset_type', type=str)
    parser.add_argument('--audios_dir', type=str)
    parser.add_argument('--output_embs_dir', type=str)
    parser.add_argument('--device', type=str)

    args = parser.parse_args()
    
    calculate_embeddings(args)