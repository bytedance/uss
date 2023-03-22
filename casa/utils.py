import os
import logging
import yaml
import datetime
import pickle
import sys

import librosa
import torch
import numpy as np
from panns_inference.models import Cnn14, Cnn14_DecisionLevelMax
from panns_inference.models import Cnn14
# from openunmix.filtering import wiener
# from audioset_source_separation.pann.models import Cnn14_DecisionLevelMax


def create_logging(log_dir, filemode):
    os.makedirs(log_dir, exist_ok=True)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, "{:04d}.log".format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, "{:04d}.log".format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        filename=log_path,
        filemode=filemode,
    )

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    return logging


def float32_to_int16(x):
    x = np.clip(x, a_min=-1, a_max=1)
    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def read_yaml(config_yaml):
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


# def str_to_class(classname):
#     return getattr(sys.modules[__name__], classname)


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pkl'.format(
            os.path.splitext(self.statistics_path)[0], 
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'balanced_train': [], 'test': [], 'valid': []}

    def append(self, step, statistics, data_type):
        statistics['step'] = step
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        logging.info('    Dump statistics to {}'.format(self.statistics_path))
        logging.info('    Dump statistics to {}'.format(self.backup_statistics_path))
        
    def load_state_dict(self, resume_step):
        self.statistics_dict = pickle.load(open(self.statistics_path, 'rb'))

        resume_statistics_dict = {'balanced_train': [], 'test': [], 'valid': []}
        
        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics['step'] <= resume_step:
                    resume_statistics_dict[key].append(statistics)
                
        self.statistics_dict = resume_statistics_dict


def load_pretrained_sed_model(sed_checkpoint_path):
    r"""Load pretrained sound event detection model.

    Args:
        sed_checkpoint_path: str

    Returns:
        sed_model: nn.Module
    """
    sed_model = Cnn14_DecisionLevelMax(sample_rate=32000, window_size=1024, 
        hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527, interpolate_mode='nearest')

    sed_checkpoint = torch.load(sed_checkpoint_path, map_location='cpu') 
    sed_model.load_state_dict(sed_checkpoint['model'])

    return sed_model


def load_pretrained_at_model(at_checkpoint_path):
    r"""Load pretrained audio tagging model.

    Args:
        at_checkpoint_path: str

    Returns:
        at_model: nn.Module
    """
    at_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, 
        mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    
    at_checkpoint = torch.load(at_checkpoint_path, map_location='cpu')
    at_model.load_state_dict(at_checkpoint['model'])

    return at_model


def energy(x):
    return torch.mean(x ** 2)


def magnitude_to_db(x):
    eps = 1e-10
    return 20. * np.log10(max(x, eps))


def db_to_magnitude(x):
    return 10. ** (x / 20)


def ids_to_hots(ids, classes_num, device):
    hots = torch.zeros(classes_num).to(device)
    for id in ids:
        hots[id] = 1
    return hots

'''
def calculate_sdr(ref, est):
    s_true = ref
    s_artif = est - ref
    sdr = 10. * (
        np.log10(np.clip(np.mean(s_true ** 2), 1e-8, np.inf)) \
        - np.log10(np.clip(np.mean(s_artif ** 2), 1e-8, np.inf)))
    return sdr
'''

def calculate_sdr(ref, est):
    
    eps = 1e-8

    noise = est - ref
    sdr = 10. * np.log10(np.sum(ref ** 2) / (np.sum(noise ** 2) + eps))

    return sdr


def calculate_sisdr(ref, est):
    
    eps = 1e-8
    ref *= np.sum(ref * est) / np.sum(ref ** 2)

    noise = est - ref
    sdr = 10. * np.log10(np.sum(ref ** 2) / (np.sum(noise ** 2) + eps) + eps)

    return sdr


def calculate_sdr_segmentwise(references, estimates, win, hop):
    pointer = 0
    sdrs = []
    while pointer + win < references.shape[-1]:
        sdr = calculate_sdr(
            ref=references[pointer : pointer + win],
            est=estimates[pointer : pointer + win],
        )
        sdrs.append(sdr)
        pointer += hop
    return sdrs
    

def get_energy_ratio(segment1, segment2):

    def _energy(x):
        return np.mean(x ** 2)

    energy1 = _energy(segment1)
    energy2 = max(1e-10, _energy(segment2))
    ratio = (energy1 / energy2) ** 0.5
    ratio = np.clip(ratio, 0.02, 50)
    return ratio


def fix_length(audio, segment_samples):
    repeats_num = (segment_samples // audio.shape[-1]) + 1
    audio = np.tile(audio, repeats_num)[0 : segment_samples]
    return audio

'''
def do_wiener(mixture, output_dict):

    source_types = output_dict.keys()

    tmp = []
    for source_type in source_types:
        stft_matrix = librosa.core.stft(
            y=output_dict[source_type], 
            n_fft=2048, 
            hop_length=320, 
            window='hann', 
            center=True
        )
        tmp.append(np.abs(stft_matrix))

    targets_spectrograms = np.stack(tmp, axis=-1)[:, :, None, :]
    targets_spectrograms = torch.Tensor(targets_spectrograms)

    mixture_stft = librosa.core.stft(
        y=mixture, 
        n_fft=2048, 
        hop_length=320, 
        window='hann', 
        center=True
    )

    mix_stft = np.stack((np.real(mixture_stft), np.imag(mixture_stft)), axis=-1)[:, :, None, :]
    mix_stft = torch.Tensor(mix_stft)

    y_stft = wiener(
        targets_spectrograms=targets_spectrograms,
        mix_stft=mix_stft,
        iterations=1,
        softmask=True,
        residual=False,
        scale_factor=10.0,
        eps=1e-10,
    )
    y_stft = y_stft.data.cpu().numpy()

    wiener_output_dict = {}
    for i, source_type in enumerate(source_types):
        tmp = y_stft[:, :, :, 0, i] + 1j * y_stft[:, :, :, 1, i]
        tmp = tmp[:, :, 0]

        recovered_audio = librosa.core.istft(
            tmp, 
            hop_length=320, 
            window='hann',
            center=True, 
            dtype=np.float32, 
            length=mixture.shape[-1],
        )
        wiener_output_dict[source_type] = recovered_audio

    return wiener_output_dict
'''