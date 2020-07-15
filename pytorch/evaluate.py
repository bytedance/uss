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


def calculate_sdr(ref, est, scaling=False):
    if scaling:
        alpha = np.dot(est, ref) / np.clip(np.dot(ref, ref), 1e-8, np.inf)
    else:
        alpha = 1.

    # s_true = ref
    # s_artif = est - ref
    s_true = alpha * ref
    s_artif = s_true - est
    sdr = 10. * (
        np.log10(np.clip(np.mean(s_true ** 2), 1e-8, np.inf)) \
        - np.log10(np.clip(np.mean(s_artif ** 2), 1e-8, np.inf)))
    return sdr


class Evaluator(object):
    def __init__(self, sed_mix, ss_model):
        # self.generator = generator
        self.sed_mix = sed_mix
        self.ss_model = ss_model
        self.device = next(ss_model.parameters()).device
        self.classes_num = config.classes_num
         
    # def evaluate(self):
    #     result_dict = self.calculate_result_dict()
    #     statistics = {key: self.average_metric(result_dict[key]) for key in result_dict.keys()}
    #     return statistics
         
    def evaluate(self, generator):

        sdr_dict = {k: [] for k in range(self.classes_num)}
        norm_sdr_dict = {k: [] for k in range(self.classes_num)}
 
        for iteration, batch_10s_dict in enumerate(generator):
            if iteration % 10 == 0:
                print(iteration)

            audios_num = len(batch_10s_dict['audio_name'])

            # Get mixture and target data
            batch_data_dict = self.sed_mix.get_mix_data(batch_10s_dict)

            # Move data to device
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], self.device)

            # Separate
            with torch.no_grad():
                self.ss_model.eval()

                batch_output_dict = self.ss_model(
                    batch_data_dict['mixture'], 
                    batch_data_dict['hard_condition'])

                batch_sep_wavs = batch_output_dict['wav'].data.cpu().numpy()            

            for n in range(0, audios_num):
                sdr = calculate_sdr(batch_data_dict['source'].data.cpu().numpy()[n, :, 0], batch_sep_wavs[n, :, 0], scaling=True)
                # norm_sdr = sdr - calculate_sdr(
                #     batch_data_dict['source'].data.cpu().numpy()[n, :, 0], 
                #     batch_data_dict['mixture'].data.cpu().numpy()[n, :, 0], scaling=True)

                class_id = batch_data_dict['class_id'].data.cpu().numpy()[n]
                sdr_dict[class_id].append(sdr)
                # norm_sdr_dict[class_id].append(norm_sdr)

            # import crash
            # asdf
            # librosa.output.write_wav('_zz.wav', batch_data_dict['source'].data.cpu().numpy()[n, :, 0], sr=32000)
            # librosa.output.write_wav('_zz2.wav', batch_data_dict['mixture'].data.cpu().numpy()[n, :, 0], sr=32000)
            # librosa.output.write_wav('_zz3.wav', batch_sep_wavs[n, :, 0], sr=32000)
            # calculate_sdr(batch_data_dict['source'].data.cpu().numpy()[n, :, 0], batch_data_dict['mixture'].data.cpu().numpy()[n, :, 0])

        # result_dict = {'sdr': sdr_dict, 'norm_sdr': norm_sdr_dict}
        result_dict = {'sdr': sdr_dict}
        return result_dict

def average_dict(metric_dict):
    values = []
    for key in metric_dict:
        if len(metric_dict[key]) > 0:
            values.append(np.mean(metric_dict[key]))

    return np.mean(values)