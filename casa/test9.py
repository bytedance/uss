import os
import re
import librosa
import time

from casa.utils import calculate_sdr
from casa.utils import create_logging, read_yaml, load_pretrained_model #, load_pretrained_sed_model, 

from casa.data.anchor_segment_detectors import AnchorSegmentDetector
from casa.data.anchor_segment_mixers import AnchorSegmentMixer
from casa.data.query_condition_extractors import QueryConditionExtractor
from casa.models.pl_modules import LitSeparation
from casa.models.resunet import *

from torch.utils.tensorboard import SummaryWriter


class AudiosetEvaluator:
    def __init__(self, pl_model, audios_dir, classes_num, max_eval_per_class=None):

        self.pl_model = pl_model
        self.audios_dir = audios_dir
        self.classes_num = classes_num
        self.device = pl_model.device
        self.max_eval_per_class = max_eval_per_class

    @torch.no_grad()
    def __call__(self):

        sdrs_dict = {class_id: [] for class_id in range(self.classes_num)}
        sdris_dict = {class_id: [] for class_id in range(self.classes_num)}

        for class_id in range(self.classes_num):

            sub_dir = os.path.join(self.audios_dir, "classid={}".format(class_id))            

            audio_names = self._get_audio_names(audios_dir=sub_dir)

            for audio_index, audio_name in enumerate(audio_names):

                source_path = os.path.join(sub_dir, "{},source.wav".format(audio_name))
                mixture_path = os.path.join(sub_dir, "{},mixture.wav".format(audio_name))

                source, fs = librosa.load(source_path, sr=None, mono=True)
                mixture, fs = librosa.load(mixture_path, sr=None, mono=True)

                sdr_no_sep = calculate_sdr(ref=source, est=mixture)
                
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

                sdr = calculate_sdr(ref=source, est=sep_segment)
                sdri = sdr - sdr_no_sep

                sdrs_dict[class_id].append(sdr)
                sdris_dict[class_id].append(sdri)

                if audio_index == self.max_eval_per_class:
                    break

            print("Class ID: {} / {}, SDR: {:.3f}, SDRi: {:.3f}".format(class_id, self.classes_num, np.mean(sdrs_dict[class_id]), np.mean(sdris_dict[class_id])))

        stats_dict = {
            "sdrs_dict": sdrs_dict,
            "sdris_dict": sdris_dict,
        }

        return stats_dict

    def _get_audio_names(self, audios_dir):
            
        audio_names = sorted(os.listdir(audios_dir))

        audio_names = [re.search('(.*),(mixture|source).wav', audio_name).group(1) for audio_name in audio_names]

        audio_names = sorted(list(set(audio_names)))

        return audio_names


def add2():

    config_yaml = "./scripts/train/01.yaml"
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
    ).to(device)

    audios_dir = "/home/tiger/workspaces/casa/evaluation/audioset/mixtures_sources_test"

    evaluator = AudiosetEvaluator(pl_model=pl_model, audios_dir=audios_dir, classes_num=classes_num, max_eval_per_class=5)

    evaluator()


def add3():
    
    writer = SummaryWriter(log_dir="_tmp/exp1")

    for i in range(10):
        writer.add_scalar('Loss/train', global_step=i, scalar_value=i * 2)
        writer.add_scalar('Loss/test', global_step=i, scalar_value=i * 3)
        


def add4(): 
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator('/home/tiger/my_code_2019.12-/python/audioset_source_separation/workspaces/casa/tf_logs/train/config=01a,devices=1/events.out.tfevents.1681729002.n130-020-141.1680684.0')

    ea.Reload()
    tags = ea.Tags()
    # values = ea.Scalars('SDRi/test')

    from IPython import embed; embed(using=False); os._exit(0)
    


if __name__ == '__main__':

    add4()