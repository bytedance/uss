import os
import re
import librosa
import time
import pathlib
import pickle

from casa.utils import calculate_sdr
from casa.utils import create_logging, read_yaml, load_pretrained_model #, load_pretrained_sed_model, 

from casa.data.anchor_segment_detectors import AnchorSegmentDetector
from casa.data.anchor_segment_mixers import AnchorSegmentMixer
from casa.data.query_condition_extractors import QueryConditionExtractor
from casa.models.pl_modules import LitSeparation
from casa.models.resunet import *
from casa.config import IX_TO_LB


def add():

    config_yaml = "./scripts/train/01b.yaml"
    sample_rate = 32000
    classes_num = 527
    device = 'cuda'

    configs = read_yaml(config_yaml)

    num_workers = configs['train']['num_workers']
    model_type = configs['model']['model_type']
    input_channels = configs['model']['input_channels']
    output_channels = configs['model']['output_channels']
    condition_size = configs['data']['condition_size']
    loss_type = configs['train']['loss_type']
    learning_rate = float(configs['train']['learning_rate'])
    condition_type = configs['data']['condition_type']

    sample_rate = configs['data']['sample_rate']

    at_model = load_pretrained_model(
        model_name=configs['audio_tagging']['model_name'],
        checkpoint_path=configs['audio_tagging']['checkpoint_path'],
        freeze=configs['audio_tagging']['freeze'],
    )

    query_condition_extractor = QueryConditionExtractor(
        at_model=at_model,
        condition_type=condition_type,
    )

    Model = eval(model_type)
    # Model = str_to_class(model_type)

    ss_model = Model(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
    )

    # pytorch-lightning model
    pl_model = LitSeparation(
        ss_model=ss_model,
        anchor_segment_detector=None,
        anchor_segment_mixer=None,
        query_condition_extractor=query_condition_extractor,
        loss_function=None,
        learning_rate=None,
        lr_lambda=None,
    )

    # checkpoint_path = "/home/tiger/my_code_2019.12-/python/audioset_source_separation/lightning_logs/version_9/checkpoints/step=80000.ckpt"

    # for step in [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000]:
    for step in [20000, 40000, 60000, 80000, 100000]:
        # checkpoint_path = "/home/tiger/my_code_2019.12-/python/audioset_source_separation/lightning_logs/version_9/checkpoints/step={}.ckpt".format(step)
        checkpoint_path = "/home/tiger/workspaces/casa/checkpoints/train/config={},devices=1/step={}.ckpt".format(pathlib.Path(config_yaml).stem, step)

        pl_model = LitSeparation.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            strict=False,
            ss_model=ss_model,
            anchor_segment_detector=None,
            anchor_segment_mixer=None,
            query_condition_extractor=query_condition_extractor,
            loss_function=None,
            learning_rate=None,
            lr_lambda=None,
        ).to(device)

        audios_dir = "/home/tiger/workspaces/casa/evaluation/audioset/mixtures_sources_test"

        evaluator = AudioSetEvaluator(pl_model=pl_model, audios_dir=audios_dir, classes_num=classes_num, max_eval_per_class=10)

        stats_dict = evaluator()

        mean_sdris = {}

        for class_id in range(classes_num):
            mean_sdris[class_id] = np.nanmean(stats_dict['sdris_dict'][class_id])
            print("{} {}: {:.3f}".format(class_id, IX_TO_LB[class_id], mean_sdris[class_id]))

        final_sdri = np.nanmean([mean_sdris[class_id] for class_id in range(classes_num)])
        print("--------")
        print("Final avg SDRi: {:.3f}".format(final_sdri))

        
        stat_path = os.path.join("stats", pathlib.Path(config_yaml).stem, "step={}.pkl".format(step))
        os.makedirs(os.path.dirname(stat_path), exist_ok=True)
        pickle.dump(stats_dict, open(stat_path, 'wb'))
        print("Write out to {}".format(stat_path))
        

if __name__ == '__main__':

    add()