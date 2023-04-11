import os
import logging
import h5py
import soundfile
import librosa
import numpy as np
import pandas as pd
from scipy import stats 
import datetime
import pickle


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def get_sub_filepaths(folder):
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            paths.append(path)
    return paths
    
    
def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging


def read_metadata(csv_path, classes_num, id_to_ix):
    """Read metadata of AudioSet from a csv file.
    Args:
      csv_path: str
    Returns:
      meta_dict: {'audio_name': (audios_num,), 'target': (audios_num, classes_num)}
    """

    with open(csv_path, 'r') as fr:
        lines = fr.readlines()
        lines = lines[3:]   # Remove heads

    audios_num = len(lines)
    targets = np.zeros((audios_num, classes_num), dtype=np.bool)
    audio_names = []
 
    for n, line in enumerate(lines):
        items = line.split(', ')
        """items: ['--4gqARaEJE', '0.000', '10.000', '"/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"\n']"""

        audio_name = 'Y{}.wav'.format(items[0])   # Audios are started with an extra 'Y' when downloading
        label_ids = items[3].split('"')[1].split(',')

        audio_names.append(audio_name)

        # Target
        for id in label_ids:
            ix = id_to_ix[id]
            targets[n, ix] = 1
    
    meta_dict = {'audio_name': np.array(audio_names), 'target': targets}
    return meta_dict


def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.2
    x = np.clip(x, -1, 1)
    return (x * 32767.).astype(np.int16)

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)
    

def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x[0 : audio_length]

