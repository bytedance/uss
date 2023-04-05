import os
import re
import librosa

from casa.utils import calculate_sdr
from casa.utils import create_logging, read_yaml, load_pretrained_model #, load_pretrained_sed_model, 

from casa.data.anchor_segment_detectors import AnchorSegmentDetector
from casa.data.anchor_segment_mixers import AnchorSegmentMixer
from casa.data.query_condition_extractors import QueryConditionExtractor
from casa.models.pl_modules import LitSeparation
from casa.models.resunet import *


class Eva:
    def __init__(self, pl_model):
        self.pl_model = pl_model

    def __call__(self):

        import numpy as np
        import torch
        tmp = np.zeros((1, 1, 32000 * 2))
        tmp = torch.Tensor(tmp).to('cuda')

        cond = np.zeros((1, 2048))
        cond = torch.Tensor(cond).to('cuda')

        input_dict = {'mixture': tmp, 'condition': cond}

        output_dict = self.pl_model.ss_model(input_dict)
        sep_wav = output_dict['waveform'].data.cpu().numpy().squeeze()

        from IPython import embed; embed(using=False); os._exit(0)


class AA:
    def __init__(self, pl_model, audios_dir):

        self.pl_model = pl_model
        self.audios_dir = audios_dir
        self.device = pl_model.device

        audio_names = sorted(os.listdir(audios_dir))

        audio_names = [re.search('(.*),(mixture|source).wav', audio_name).group(1) for audio_name in audio_names]

        self.audio_names = sorted(list(set(audio_names)))

    @torch.no_grad()
    def __call__(self):

        for audio_name in self.audio_names:

            source_path = os.path.join(self.audios_dir, "{},source.wav".format(audio_name))
            mixture_path = os.path.join(self.audios_dir, "{},mixture.wav".format(audio_name))

            source, fs = librosa.load(source_path, sr=None, mono=True)
            mixture, fs = librosa.load(mixture_path, sr=None, mono=True)

            calculate_sdr(ref=source, est=mixture)

            # conditions = self.query_condition_extractor(
            #     segments=segments,
            # )
            
            conditions = self.pl_model.query_condition_extractor(
                segments=torch.Tensor(source)[None, :].to(self.device),
            )

            input_dict = {
                'mixture': torch.Tensor(mixture)[None, None, :].to(self.device),
                'condition': conditions,
            }

            self.pl_model.eval()
            sep_segment = self.pl_model.ss_model(input_dict)['waveform']
            sep_segment = sep_segment.squeeze().data.cpu().numpy()

            calculate_sdr(ref=source, est=sep_segment)

            from IPython import embed; embed(using=False); os._exit(0)

            


def add():

    audios_dir = "/home/tiger/workspaces/casa/evaluation/audioset/mixtures_sources_test"

    at_model = load_pretrained_model(
        model_name=configs['audio_tagging']['model_name'],
        checkpoint_path=configs['audio_tagging']['checkpoint_path'],
        freeze=configs['audio_tagging']['freeze'],
    )

    query_condition_extractor = QueryConditionExtractor(
        at_model=at_model,
        condition_type='embedding',
    )

    eva = AA(audios_dir=audios_dir)

    eva()


def add2():

    config_yaml = "./scripts/train/01.yaml"
    sample_rate = 32000

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

    save_step_frequency = configs['train']['save_step_frequency']

    sed_model = load_pretrained_model(
        model_name=configs['sound_event_detection']['model_name'],
        checkpoint_path=configs['sound_event_detection']['checkpoint_path'],
        freeze=configs['sound_event_detection']['freeze'],
    )

    at_model = load_pretrained_model(
        model_name=configs['audio_tagging']['model_name'],
        checkpoint_path=configs['audio_tagging']['checkpoint_path'],
        freeze=configs['audio_tagging']['freeze'],
    )

    anchor_segment_detector = AnchorSegmentDetector(
        sed_model=sed_model,
        clip_seconds=10.,
        segment_seconds=2.,
        frames_per_second=100,
        sample_rate=sample_rate,
    )

    query_condition_extractor = QueryConditionExtractor(
        at_model=at_model,
        condition_type='embedding',
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

    checkpoint_path = "/home/tiger/my_code_2019.12-/python/audioset_source_separation/lightning_logs/version_0/checkpoints/lightning_logs/version_8/checkpoints/step=1.ckpt"

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
    )

    audios_dir = "/home/tiger/workspaces/casa/evaluation/audioset/mixtures_sources_test"

    eva = AA(pl_model=pl_model, audios_dir=audios_dir)

    eva()

if __name__ == '__main__':

    add2()