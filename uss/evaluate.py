import logging
import os
import pathlib
import re
from typing import Dict, List

import librosa
import lightning.pytorch as pl
import numpy as np
import torch

from uss.config import IX_TO_LB
from uss.inference import load_ss_model
from uss.utils import (calculate_sdr, create_logging, get_mean_sdr_from_dict,
                       parse_yaml)


class AudioSetEvaluator:
    def __init__(
        self,
        audios_dir: str,
        classes_num: int,
        max_eval_per_class=None,
    ) -> None:
        r"""AudioSet evaluator.

        Args:
            audios_dir (str): directory of evaluation segments
            classes_num (int): the number of sound classes
            max_eval_per_class (int), the number of samples to evaluate for each sound class

        Returns:
            None
        """

        self.audios_dir = audios_dir
        self.classes_num = classes_num
        self.max_eval_per_class = max_eval_per_class

    @torch.no_grad()
    def __call__(
        self,
        pl_model: pl.LightningModule
    ) -> Dict:
        r"""Evalute."""

        sdrs_dict = {class_id: [] for class_id in range(self.classes_num)}
        sdris_dict = {class_id: [] for class_id in range(self.classes_num)}

        for class_id in range(self.classes_num):

            sub_dir = os.path.join(
                self.audios_dir,
                "class_id={}".format(class_id))

            audio_names = self._get_audio_names(audios_dir=sub_dir)

            for audio_index, audio_name in enumerate(audio_names):

                if audio_index == self.max_eval_per_class:
                    break

                source_path = os.path.join(
                    sub_dir, "{},source.wav".format(audio_name))
                mixture_path = os.path.join(
                    sub_dir, "{},mixture.wav".format(audio_name))

                source, fs = librosa.load(source_path, sr=None, mono=True)
                mixture, fs = librosa.load(mixture_path, sr=None, mono=True)

                sdr_no_sep = calculate_sdr(ref=source, est=mixture)

                device = pl_model.device

                conditions = pl_model.query_net(
                    source=torch.Tensor(source)[None, :].to(device),
                )["output"]
                # conditions: (batch_size=1, condition_dim)

                input_dict = {
                    "mixture": torch.Tensor(mixture)[None, None, :].to(device),
                    "condition": conditions,
                }

                pl_model.eval()
                sep_segment = pl_model.ss_model(input_dict)["waveform"]
                # sep_segment: (batch_size=1, channels_num=1, segment_samples)

                sep_segment = sep_segment.squeeze(
                    dim=(0, 1)).data.cpu().numpy()
                # sep_segment: (segment_samples,)

                sdr = calculate_sdr(ref=source, est=sep_segment)
                sdri = sdr - sdr_no_sep

                sdrs_dict[class_id].append(sdr)
                sdris_dict[class_id].append(sdri)

            logging.info(
                "Class ID: {} / {}, SDR: {:.3f}, SDRi: {:.3f}".format(
                    class_id, self.classes_num, np.mean(
                        sdrs_dict[class_id]), np.mean(
                        sdris_dict[class_id])))

        stats_dict = {
            "sdrs_dict": sdrs_dict,
            "sdris_dict": sdris_dict,
        }

        return stats_dict

    def _get_audio_names(self, audios_dir: str) -> List[str]:
        r"""Get evaluation audio names."""

        audio_names = sorted(os.listdir(audios_dir))

        audio_names = [
            re.search(
                "(.*),(mixture|source).wav",
                audio_name).group(1) for audio_name in audio_names]

        audio_names = sorted(list(set(audio_names)))

        return audio_names

    @staticmethod
    def get_median_metrics(stats_dict, metric_type):
        class_ids = stats_dict[metric_type].keys()
        median_stats_dict = {
            class_id: np.nanmedian(
                stats_dict[metric_type][class_id]) for class_id in class_ids}
        return median_stats_dict


def test_evaluate(config_yaml: str, workspace: str):
    r"""Evaluate using pretrained checkpoint.

    Args:
        config_yaml (str), path of the config yaml file
        workspace (str), directory of workspace

    Returns:
        None
    """

    device = "cuda"

    configs = parse_yaml(config_yaml)

    classes_num = configs["data"]["classes_num"]

    audios_dir = os.path.join(
        workspace, "evaluation/audioset/2s_segments_test")

    create_logging("_tmp_log", filemode="w")

    # Evlauator
    evaluator = AudioSetEvaluator(
        audios_dir=audios_dir,
        classes_num=classes_num,
        max_eval_per_class=10
    )

    steps = [1]

    for step in steps:

        # Checkpoint path
        checkpoint_path = os.path.join(
            workspace, "checkpoints/train/config={},devices=1/step={}.ckpt".format(
                pathlib.Path(config_yaml).stem, step))

        # Load model
        pl_model = load_ss_model(
            configs=configs,
            checkpoint_path=checkpoint_path
        ).to(device)

        # Evaluate statistics
        stats_dict = evaluator(pl_model=pl_model)

        median_sdris = {}

        for class_id in range(classes_num):

            median_sdris[class_id] = np.nanmedian(
                stats_dict["sdris_dict"][class_id])

            print(
                "{} {}: {:.3f}".format(
                    class_id,
                    IX_TO_LB[class_id],
                    median_sdris[class_id]))

        mean_sdri = get_mean_sdr_from_dict(median_sdris)
        # final_sdri = np.nanmean([mean_sdris[class_id]
        # for class_id in range(classes_num)])
        print("--------")
        print("Average SDRi: {:.3f}".format(mean_sdri))


if __name__ == "__main__":

    test_evaluate(
        config_yaml="./scripts/train/ss_model=resunet30,querynet=at_soft,gpus=1.yaml",
        workspace="./workspaces/uss",
    )
