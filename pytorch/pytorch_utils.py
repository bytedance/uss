import numpy as np
import librosa
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from utilities import create_folder
import config


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_average_energy(x):
    return np.mean(np.square(x))


def id_to_one_hot(id, classes_num):
    one_hot = np.zeros(classes_num)
    one_hot[id] = 1
    return one_hot


class SedMix(object):
    def __init__(self, sed_model, neighbour_segs, sample_rate):
        self.sed_model = sed_model
        self.neighbour_segs = neighbour_segs
        self.sample_rate = sample_rate
        self.device = next(sed_model.parameters()).device
        self.seg_hop_sec = sed_model.module.segment_hop_sec
        self.random_state = np.random.RandomState(1234)

    def get_sample_bgn_fin_indexes(self, seg_index, neighbour_segs, total_segs_num, seg_hop_sec):
        bgn_seg_idx = seg_index - neighbour_segs
        fin_seg_idx = seg_index + neighbour_segs + 1
        if bgn_seg_idx < 0:
            bgn_seg_idx = 0
            fin_seg_idx = neighbour_segs * 2 + 1
        if fin_seg_idx > total_segs_num:
            bgn_seg_idx = total_segs_num - (neighbour_segs * 2 + 1)
            fin_seg_idx = total_segs_num
        bgn_sample_idx = int((seg_hop_sec * self.sample_rate) * bgn_seg_idx)
        fin_sample_idx = int((seg_hop_sec * self.sample_rate) * fin_seg_idx)
        return bgn_sample_idx, fin_sample_idx


    def get_mix_data(self, data_dict, sed_model, with_identity_zero): 
        (audios_num, classes_num) = data_dict['target'].shape

        batch_waveform = move_data_to_device(data_dict['waveform'], self.device)

        # SED
        with torch.no_grad():
            sed_model.eval()
            _output_dict = sed_model(batch_waveform)
            # clipwise_output = output_dict['clipwise_output'].data.cpu().numpy()
            segmentwise_output = _output_dict['segmentwise_output'].data.cpu().numpy()

        total_segs_num = segmentwise_output.shape[1]

        # Get segment with maximum prediction for a sound class
        # seg_predictions = []
        seg_waveforms = []
        for n in range(audios_num):
            seg_index = np.argmax(segmentwise_output[n, :, data_dict['class_id'][n]])
            # seg_predictions.append(segmentwise_output[n, seg_index, :])
            (bgn_sample_idx, fin_sample_idx) = self.get_sample_bgn_fin_indexes(
                seg_index, self.neighbour_segs, total_segs_num, self.seg_hop_sec)
            seg_waveforms.append(data_dict['waveform'][n, bgn_sample_idx : fin_sample_idx])

        seg_waveforms = np.array(seg_waveforms)

        # Predict tags of extracted segments
        with torch.no_grad():
            sed_model.eval()
            _output_dict = sed_model(move_data_to_device(seg_waveforms, self.device))
            seg_predictions = _output_dict['clipwise_output'].data.cpu().numpy()

        mixtures = []
        sources = []
        soft_conditions = []
        hard_conditions = []
        for n in range(0, audios_num, 2):
            ratio = (calculate_average_energy(seg_waveforms[n]) / max(1e-8, calculate_average_energy(seg_waveforms[n + 1]))) ** 0.5
            ratio = np.clip(ratio, 0.02, 50)
            seg_waveforms[n + 1] *= ratio
            mixture = seg_waveforms[n] + seg_waveforms[n + 1]

            # Mixutres
            mixtures.append(mixture)
            mixtures.append(mixture)
            _rnd = np.random.randint(2)
            m = n + _rnd
            m2 = n + (1 - _rnd)
            mixtures.append(seg_waveforms[m])
            mixtures.append(seg_waveforms[m])

            # Targets
            sources.append(seg_waveforms[n])
            sources.append(seg_waveforms[n + 1])
            sources.append(seg_waveforms[m])
            sources.append(np.zeros_like(seg_waveforms[m]))
            
            # Soft conditions
            soft_conditions.append(seg_predictions[n])
            soft_conditions.append(seg_predictions[n + 1])
            soft_conditions.append(seg_predictions[m])

            # f(x1, c2) -> 0. Make sure c2 and the prediction of x1 is exclusive. 
            for k in range(classes_num):
                if seg_predictions[m, k] >= 0.02:
                    seg_predictions[m2, k] = 0.
            soft_conditions.append(seg_predictions[m2])
            
            # Hard conditions
            hard_conditions.append(id_to_one_hot(data_dict['class_id'][n], classes_num))
            hard_conditions.append(id_to_one_hot(data_dict['class_id'][n + 1], classes_num))
            hard_conditions.append(id_to_one_hot(data_dict['class_id'][m], classes_num))
            hard_conditions.append(id_to_one_hot(data_dict['class_id'][m2], classes_num))
                
        if with_identity_zero:
            indexes = np.arange(len(mixtures))  # (0, 1, ..., batch_size)
        else:
            N = len(mixtures)
            indexes = np.arange(N).reshape((N // 4, 4))[:, 0 : 2].flatten()
            """(0, 1, 4, 5, 8, 9, ..., batch_size * 2 - 4, batch_size * 2 - 3)"""

        output_dict = {
            'class_id': data_dict['class_id'],  # (batch_size,)
            # 'clipwise_output': clipwise_output, # (batch_size,)
            'segmentwise_output': segmentwise_output,   # (batch_size,)
            'mixture': np.array(mixtures)[indexes], # (batch_size,) | (batch_size * 2,)
            'source': np.array(sources)[indexes],   # (batch_size,) | (batch_size * 2,)
            'soft_condition': np.array(soft_conditions)[indexes],   # (batch_size,) | (batch_size * 2,)
            'hard_condition': np.array(hard_conditions)[indexes],   # (batch_size,) | (batch_size * 2,)
            }

        return output_dict


def debug_and_plot(ss_model, batch_10s_dict, batch_data_dict):
    sample_rate = config.sample_rate
    ix_to_lb = config.ix_to_lb
    device = next(ss_model.parameters()).device
    
    create_folder('_debug')

    i = 44 # 4, 20, 21, 23, 28, 34, 42, 44*
    print('Audio: {}, major label: {}'.format(i, ix_to_lb[batch_data_dict['class_id'][i * 2]]))
    print('Audio: {}, major label: {}'.format(i, ix_to_lb[batch_data_dict['class_id'][i * 2 + 1]]))
    librosa.output.write_wav('_debug/10s_0.wav', batch_10s_dict['waveform'][i * 2], sr=sample_rate)
    librosa.output.write_wav('_debug/10s_1.wav', batch_10s_dict['waveform'][i * 2 + 1], sr=sample_rate)
    librosa.output.write_wav('_debug/src_0.wav', batch_data_dict['source'][i * 4], sr=sample_rate)
    librosa.output.write_wav('_debug/src_1.wav', batch_data_dict['source'][i * 4 + 1], sr=sample_rate)
    librosa.output.write_wav('_debug/src_2.wav', batch_data_dict['source'][i * 4 + 2], sr=sample_rate)
    librosa.output.write_wav('_debug/src_3.wav', batch_data_dict['source'][i * 4 + 3], sr=sample_rate)
    librosa.output.write_wav('_debug/mix.wav', batch_data_dict['mixture'][i * 4], sr=sample_rate)

    spec1 = ss_model.module.wavin_to_specin(torch.Tensor(batch_10s_dict['waveform'][i * 2][None, :]).to(device)).data.cpu().numpy()[0, 0]
    spec2 = ss_model.module.wavin_to_specin(torch.Tensor(batch_10s_dict['waveform'][i * 2 + 1][None, :]).to(device)).data.cpu().numpy()[0, 0]

    fig, axs = plt.subplots(4,1, sharex=False)
    # axs[0].matshow(np.log(spec1).T, origin='lower', aspect='auto', cmap='jet')
    axs[0].plot(batch_10s_dict['waveform'][i * 2])
    axs[1].plot(batch_data_dict['segmentwise_output'][i * 2, :, batch_data_dict['class_id'][i * 2]])
    axs[2].matshow(np.log(spec2).T, origin='lower', aspect='auto', cmap='jet')
    axs[3].plot(batch_data_dict['segmentwise_output'][i * 2 + 1, :, batch_data_dict['class_id'][i * 2 + 1]])
    axs[0].set_title(ix_to_lb[batch_data_dict['class_id'][i * 2]])
    axs[2].set_title(ix_to_lb[batch_data_dict['class_id'][i * 2 + 1]])
    plt.savefig('_debug/_spectrogram.pdf')
    print('Saved debug info to _debug/')

    import crash
    asdf