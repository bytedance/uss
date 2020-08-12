import os
import sys
import librosa
sys.path.insert(1, os.path.join(sys.path[0], '../pytorch'))
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import csv
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle
import musdb
import museval

import torch
import torch.optim as optim
# from sed_models import Cnn13_DecisionLevelMax
from models import UNet
from utilities import (create_folder, get_filename, create_logging, StatisticsContainer)
import config
from losses import get_loss_func
from data_generator import SsAudioSetDataset, SsBalancedSampler, SsEvaluateSampler, collect_fn 
# from pytorch_utils import count_parameters, SedMix, debug_and_plot, move_data_to_device
from pytorch_utils import count_parameters, SedMix, move_data_to_device, id_to_one_hot
from panns_inference import SoundEventDetection, AudioTagging, labels
from evaluate import Evaluator, average_dict, calculate_sdr


def read_musdb18_csv(csv_path):
    """Read audio names from csv file.

    Args:
      csv_path: str

    Returns:
      audio_names: list of str, e.g. ['A Classic Education - NightOwl', 
        'ANiMAL - Clinic A', 'ANiMAL - Easy Tiger', ...]
    """
    with open(csv_path, 'r') as fr:
        reader = csv.reader(fr, delimiter='\t')
        lines = list(reader)
    audio_names = [line[0] for line in lines]

    return audio_names


class SourceSeparation(object):
    def __init__(self, segment_samples=44100*3):

        sample_rate = config.sample_rate
        clip_samples = config.clip_samples
        self.classes_num = config.classes_num
        segment_frames = config.segment_frames
        self.device = 'cuda'
        model_type = 'UNet'
        self.segment_samples = segment_samples

        # Paths
        checkpoint_path = '/home/tiger/workspaces/audioset_source_separation/checkpoints/ss_main/data_type=balanced_train/UNet/loss_type=mae/balanced=balanced/augmentation=none/mix_type=5/batch_size=12/480000_iterations.pth'
        # checkpoint_path = '/home/tiger/workspaces/audioset_source_separation/checkpoints/ss_main/data_type=balanced_train/UNet/loss_type=mae/balanced=balanced/augmentation=none/mix_type=5b/batch_size=12/200000_iterations.pth'


        if 'cuda' in str(self.device):
            logging.info('Using GPU.')
            device = 'cuda'
        else:
            logging.info('Using CPU.')
            device = 'cpu'

        # Source separation model
        SsModel = eval(model_type)
        ss_model = SsModel(channels=1)
        
        # Resume training
        checkpoint = torch.load(checkpoint_path)
        ss_model.load_state_dict(checkpoint['model'])
        
        # Parallel
        print('GPU number: {}'.format(torch.cuda.device_count()))
        self.ss_model = torch.nn.DataParallel(ss_model)

        if 'cuda' in str(device):
            ss_model.to(device)
        
        # sed_checkpoint_path = '/home/tiger/released_models/sed/Cnn14_DecisionLevelMax_mAP=0.385.pth'
        # at_checkpoint_path = '/home/tiger/released_models/sed/Cnn14_mAP=0.431.pth'
        # self.sed_model = SoundEventDetection(device=device, checkpoint_path=sed_checkpoint_path)
        # self.at_model = AudioTagging(device=device, checkpoint_path=at_checkpoint_path)
        # self.sed_mix = SedMix(sed_model, at_model, segment_frames=segment_frames, sample_rate=sample_rate)

        # (clipwise_output, embedding) = at_model.inference(audio[None, :])
        # print_audio_tagging_result(clipwise_output[0])

    def separate(self, audio):
        audio = audio[None, :, :]

        if self.segment_samples is None:
            with torch.no_grad():
                audio = move_data_to_device(audio, self.device)

                self.model.eval()
                output = self.model(audio)['wav']
                output = output[0].data.cpu().numpy()

        else:
            audio_len = audio.shape[1]
            pad_len = int(np.ceil(audio_len / self.segment_samples)) * self.segment_samples - audio_len
            audio = np.concatenate((audio, np.zeros((1, pad_len, audio.shape[2]))), axis=1)

            # audio = move_data_to_device(audio, self.device)

            batch = self.enframe(audio, self.segment_samples)
            output = []

            for seg in batch:
                with torch.no_grad():
                    self.ss_model.eval()
                    id1 = 0
                    tmp = []
                    for i in range(2):
                        hard_condition = id_to_one_hot(id1, self.classes_num)[None, :]
                        batch_data_dict = {'mixture': seg[:, i][None, :, None], 'hard_condition': hard_condition}

                        # Move data to device
                        for key in batch_data_dict.keys():
                            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], self.device)
                        
                        batch_output_dict = self.ss_model(batch_data_dict['mixture'], batch_data_dict['hard_condition'])
                        tmp.append(batch_output_dict['wav'][0].data.cpu().numpy())

                    tmp = np.concatenate(tmp, axis=-1)

                output.append(tmp)
            
            output = np.array(output)
            output = self.deframe(output)
            output = output[0 : audio_len]

        return output

    def enframe(self, x, segment_samples):
        assert x.shape[1] % segment_samples == 0
        batch = []

        pointer = 0
        while pointer + segment_samples <= x.shape[1]:
            batch.append(x[:, pointer : pointer + segment_samples, :])
            pointer += segment_samples // 2

        batch = np.concatenate(batch, axis=0)
        return batch

    def deframe(self, x):
        if x.shape[0] == 1:
            return x[0]
        else:
            (N, segment_samples, _) = x.shape
            y = []
            y.append(x[0, 0 : int(segment_samples * 0.75)])
            for i in range(1, N - 1):
                y.append(x[i, int(segment_samples * 0.25) : int(segment_samples * 0.75)])
            y.append(x[-1, int(segment_samples * 0.25) :])
            y = np.concatenate(y, axis=0)
            return y



def evaluate(args):
    # Arugments & parameters
    workspace = args.workspace
    # at_checkpoint_path = args.at_checkpoint_path
    # data_type = args.data_type
    # model_type = args.model_type
    # loss_type = args.loss_type
    # balanced = args.balanced
    # augmentation = args.augmentation
    # batch_size = args.batch_size
    # iteration = args.iteration
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename

    num_workers = 8
    sample_rate = config.sample_rate
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    segment_frames = config.segment_frames
    # loss_func = get_loss_func(loss_type)

    # audios_dir = '/home/tiger/datasets/musdb18/dataset_root/test'
    dataset_dir = '/home/tiger/datasets/musdb18/dataset_root'
    test_csv = os.path.join(dataset_dir, 'piecenames_TEST.txt')

    test_names = read_musdb18_csv(test_csv)

    mus = musdb.DB(root=dataset_dir)

    ss = SourceSeparation(segment_samples=44100*3)

    all_sdrs = []

    for name in test_names:
        track = mus.tracks[mus.get_track_indices_by_names(name)[0]]

        mixture = librosa.core.resample(track.audio.T, track.rate, sample_rate, res_type='kaiser_fast').T
        target = track.targets['vocals'].audio

        sep = ss.separate(mixture)

        if True:
            librosa.output.write_wav('_tmp/zz.wav', sep, sr=sample_rate)
            os.system('ffmpeg -loglevel panic -y -i _tmp/zz.wav "_tmp/{}.mp3"'.format(name))
        
        sep = librosa.core.resample(sep.T, sample_rate, track.rate, res_type='kaiser_fast').T
        sep = sep[0 : len(target)]

        results = museval.evaluate(target.T, sep.T)
        _tmp = results[0]
        _tmp = _tmp[~np.isnan(_tmp)]
        sdr = np.median(_tmp)
        print('{}, sdr: {:.3f}'.format(name, sdr))

        all_sdrs.append(sdr)

    track = mus.tracks[mus.get_track_indices_by_names(name)[0]]
    print('mean sdr: {:.3f}'.format(np.mean(all_sdrs)))
    print('median sdr: {:.3f}'.format(np.median(all_sdrs)))
    
    import crash
    asdf
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_evaluate = subparsers.add_parser('evaluate')
    parser_evaluate.add_argument('--workspace', type=str, required=True)
    # parser_evaluate.add_argument('--at_checkpoint_path', type=str, required=True)
    # parser_evaluate.add_argument('--data_type', type=str, required=True)
    # parser_evaluate.add_argument('--model_type', type=str, required=True)
    # parser_evaluate.add_argument('--loss_type', type=str, required=True)
    # parser_evaluate.add_argument('--balanced', type=str, default='balanced', choices=['balanced'])
    # parser_evaluate.add_argument('--augmentation', type=str, default='mixup', choices=['none'])
    # parser_evaluate.add_argument('--batch_size', type=int, required=True)
    # parser_evaluate.add_argument('--iteration', type=int, required=True)
    parser_evaluate.add_argument('--cuda', action='store_true', default=False)
     
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'evaluate':
        evaluate(args)

    else:
        raise Exception('Error argument!')