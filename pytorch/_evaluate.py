import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import datetime
# import mir_eval
import librosa
import torch

from utilities import get_filename, create_folder
from pytorch_utils import move_data_to_device
import config


def expand_data(list_data_dict):

    classes_num = config.classes_num

    new_list_data_dict = []

    for data_dict in list_data_dict:
        for k in range(classes_num):
            if data_dict['target'][k] == 1:
                new_list_data_dict.append(data_dict)

 

class Evaluator(object):
    def __init__(self, generator, sed_mix, sed_model, ss_model, max_iteration):
        self.generator = generator
        self.sed_mix = sed_mix
        self.sed_model = sed_model
        self.ss_model = ss_model
        self.max_iteration = max_iteration
        self.device = next(ss_model.parameters()).device
        self.classes_num = config.classes_num
         
    def evaluate(self):
        result_dict = self.calculate_result_dict()
        statistics = {key: self.average_metric(result_dict[key]) for key in result_dict.keys()}
        return statistics
         
    def calculate_result_dict(self):
 
        sdr_dict = {id: [] for id in range(self.classes_num)}
        sir_dict = {id: [] for id in range(self.classes_num)}
        sar_dict = {id: [] for id in range(self.classes_num)}

        for iteration, batch_10s_dict in enumerate(self.generator):
            print(iteration)

            if iteration == self.max_iteration:
                break

            audios_num = len(batch_10s_dict['audio_name'])

            # Get mixture and target data
            batch_data_dict = self.sed_mix.get_mix_data(batch_10s_dict, self.sed_model, with_identity_zero=False)

            # Move data to device
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], self.device)

            audio_seg_length = batch_data_dict['mixture'].shape[-1]

            # Separate
            with torch.no_grad():
                self.ss_model.eval()
                batch_wav_output = self.ss_model.module.wavin_to_wavout(
                    batch_data_dict['mixture'], 
                    batch_data_dict['hard_condition'], 
                    batch_data_dict['soft_condition'], 
                    length=audio_seg_length).data.cpu().numpy()

            batch_wav_source = batch_data_dict['source'].data.cpu().numpy()
            
            for n in range(0, audios_num, 2):
                import crash
                asdf
                '''
                try:
                    (_sdrs, _sirs, _sars, _) = mir_eval.separation.bss_eval_sources(
                        batch_wav_source[n : n + 2], batch_wav_output[n : n + 2], compute_permutation=False)
                except:
                    (_sdrs, _sirs, _sars) = ([0, 0], [0, 0], [0, 0])
                
                for j in range(2):
                    id = batch_data_dict['class_id'].data.cpu().numpy()[n + j]
                    sdr_dict[id].append(_sdrs[j])
                    sir_dict[id].append(_sirs[j])
                    sar_dict[id].append(_sars[j])
                '''

        result_dict = {'sdr': sdr_dict, 'sir': sir_dict, 'sar': sar_dict}
        return result_dict

    def average_metric(self, metric_dict):
        metrics_array = []
        for id in range(self.classes_num):
            if len(metric_dict[id]) > 0:
                metrics_array.append(np.mean(metric_dict[id]))
        return np.mean(metrics_array)