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


def ids_to_hots(ids, classes_num):
    hots = np.zeros(classes_num)
    for id in ids:
        hots[id] = 1
    return hots


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SedMix(object):
    def __init__(self, sed_model, at_model, segment_frames, sample_rate):
        self.sed_model = sed_model
        self.at_model = at_model
        self.segment_frames = segment_frames
        self.sample_rate = sample_rate
        self.hop_samples = config.hop_samples
        self.clip_samples = config.clip_samples
        # self.device = next(sed_model.module.parameters()).device
        # import crash
        # asdf
        # self.seg_hop_sec = sed_model.module.segment_hop_sec
        self.random_state = np.random.RandomState(1234)

        import pickle
        self.opt_thres = pickle.load(open('opt_thres.pkl', 'rb'))

    '''
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
    '''
    def get_segment_bgn_end_samples(self, anchor_index, segment_frames):
        bgn_frame = anchor_index - segment_frames // 2
        end_frame = anchor_index + segment_frames // 2

        bgn_sample = bgn_frame * self.hop_samples
        end_sample = end_frame * self.hop_samples

        segment_samples = segment_frames * self.hop_samples

        if bgn_sample < 0:
            bgn_sample = 0
            end_sample = segment_samples

        if end_sample > self.clip_samples:
            bgn_sample = self.clip_samples - segment_samples
            end_sample = self.clip_samples

        return bgn_sample, end_sample


    def get_mix_data(self, data_dict): 

        framewise_output = self.sed_model.inference(data_dict['waveform'])
        (audios_num, total_frames_num, classes_num) = framewise_output.shape

        # total_segs_num = segmentwise_output.shape[1]

        # Get segment with maximum prediction for a sound class
        # seg_predictions = []
        seg_waveforms = []
        for n in range(audios_num):
            smoothed_framewise_output = np.convolve(
                framewise_output[n, :, data_dict['class_id'][n]], np.ones(self.segment_frames), mode='same')
            anchor_index = np.argmax(smoothed_framewise_output)

            (bgn_sample, end_sample) = self.get_segment_bgn_end_samples(
                anchor_index, self.segment_frames)

            seg_waveforms.append(data_dict['waveform'][n, bgn_sample : end_sample])

        seg_waveforms = np.array(seg_waveforms)
        seg_predictions, _ = self.at_model.inference(seg_waveforms)
        
        mixtures = []
        sources = []
        soft_conditions = []
        hard_conditions = []
        for n in range(0, audios_num, 2):
            ratio = (calculate_average_energy(seg_waveforms[n]) / max(1e-8, calculate_average_energy(seg_waveforms[n + 1]))) ** 0.5
            ratio = np.clip(ratio, 0.02, 50)
            seg_waveforms[n + 1] *= ratio
            mixture = seg_waveforms[n] + seg_waveforms[n + 1]

            if False:
                import crash
                asdf 
                K = 3
                config.ix_to_lb[data_dict['class_id'][K]]
                data_dict['target'][K]
                [config.ix_to_lb[idx] for idx in np.argsort(seg_predictions[K])[::-1][0:10]]
                librosa.output.write_wav('_zz.wav', data_dict['waveform'][K], sr=32000)
                librosa.output.write_wav('_zz2.wav', seg_waveforms[K], sr=32000)
                seg_predictions[K, data_dict['class_id'][K]]
                np.max(framewise_output[K, :, data_dict['class_id'][K]])

            # Mixutres
            mixtures.append(mixture)
            mixtures.append(mixture)
            # _rnd = np.random.randint(2)
            # m = n + _rnd
            # m2 = n + (1 - _rnd)
            # mixtures.append(seg_waveforms[m])
            # mixtures.append(seg_waveforms[m])

            # Targets
            sources.append(seg_waveforms[n])
            sources.append(seg_waveforms[n + 1])
            # sources.append(seg_waveforms[m])
            # sources.append(np.zeros_like(seg_waveforms[m]))
            
            # Soft conditions
            # soft_conditions.append(seg_predictions[n])
            # soft_conditions.append(seg_predictions[n + 1])
            # soft_conditions.append(seg_predictions[m])

            # f(x1, c2) -> 0. Make sure c2 and the prediction of x1 is exclusive. 
            # for k in range(classes_num):
            #     if seg_predictions[m, k] >= 0.02:
            #         seg_predictions[m2, k] = 0.
            # soft_conditions.append(seg_predictions[m2])
            
            # Hard conditions
            hard_conditions.append(id_to_one_hot(data_dict['class_id'][n], classes_num))
            hard_conditions.append(id_to_one_hot(data_dict['class_id'][n + 1], classes_num))
            # hard_conditions.append(id_to_one_hot(data_dict['class_id'][m], classes_num))
            # hard_conditions.append(id_to_one_hot(data_dict['class_id'][m2], classes_num))
                

        # indexes = np.arange(len(mixtures))  # (0, 1, ..., batch_size)
        # if with_identity_zero:
        #     indexes = np.arange(len(mixtures))  # (0, 1, ..., batch_size)
        # else:
        #     N = len(mixtures)
        #     indexes = np.arange(N).reshape((N // 4, 4))[:, 0 : 2].flatten()
        #     """(0, 1, 4, 5, 8, 9, ..., batch_size * 2 - 4, batch_size * 2 - 3)"""

        output_dict = {
            'class_id': data_dict['class_id'],  # (batch_size,)
            'mixture': np.array(mixtures)[:, :, None], # (batch_size,) | (batch_size * 2,)
            'source': np.array(sources)[:, :, None],   # (batch_size,) | (batch_size * 2,)
            'hard_condition': np.array(hard_conditions),   # (batch_size,) | (batch_size * 2,)
            }

        return output_dict


    def get_mix_data2(self, data_dict): 

        framewise_output = self.sed_model.inference(data_dict['waveform'])
        (audios_num, total_frames_num, classes_num) = framewise_output.shape

        seg_waveforms = []
        for n in range(audios_num):
            smoothed_framewise_output = np.convolve(
                framewise_output[n, :, data_dict['class_id'][n]], np.ones(self.segment_frames), mode='same')
            anchor_index = np.argmax(smoothed_framewise_output)

            (bgn_sample, end_sample) = self.get_segment_bgn_end_samples(
                anchor_index, self.segment_frames)

            seg_waveforms.append(data_dict['waveform'][n, bgn_sample : end_sample])

        seg_waveforms = np.array(seg_waveforms)
        seg_predictions, _ = self.at_model.inference(seg_waveforms)
        
        mixtures = []
        sources = []
        soft_conditions = []
        hard_conditions = []
        class_ids = []
        for n in range(0, audios_num, 2):
            ratio = (calculate_average_energy(seg_waveforms[n]) / max(1e-8, calculate_average_energy(seg_waveforms[n + 1]))) ** 0.5
            ratio = np.clip(ratio, 0.02, 50)
            seg_waveforms[n + 1] *= ratio
            mixture = seg_waveforms[n] + seg_waveforms[n + 1]

            if False:
                import crash
                asdf 
                K = 3
                config.ix_to_lb[data_dict['class_id'][K]]
                data_dict['target'][K]
                [config.ix_to_lb[idx] for idx in np.argsort(seg_predictions[K])[::-1][0:10]]
                librosa.output.write_wav('_zz.wav', data_dict['waveform'][K], sr=32000)
                librosa.output.write_wav('_zz2.wav', seg_waveforms[K], sr=32000)
                seg_predictions[K, data_dict['class_id'][K]]
                np.max(framewise_output[K, :, data_dict['class_id'][K]])

            # Mixutres
            mixtures.append(mixture)
            mixtures.append(mixture)
            _rnd = self.random_state.randint(2)
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
            # soft_conditions.append(seg_predictions[n])
            # soft_conditions.append(seg_predictions[n + 1])
            # soft_conditions.append(seg_predictions[m])

            # f(x1, c2) -> 0. Make sure c2 and the prediction of x1 is exclusive. 
            # for k in range(classes_num):
            #     if seg_predictions[m, k] >= 0.02:
            #         seg_predictions[m2, k] = 0.
            # soft_conditions.append(seg_predictions[m2])
            
            # Hard conditions
            hard_conditions.append(id_to_one_hot(data_dict['class_id'][n], classes_num))
            hard_conditions.append(id_to_one_hot(data_dict['class_id'][n + 1], classes_num))
            hard_conditions.append(id_to_one_hot(data_dict['class_id'][m], classes_num))
            hard_conditions.append(id_to_one_hot(data_dict['class_id'][m2], classes_num))

            class_ids.append(data_dict['class_id'][n])
            class_ids.append(data_dict['class_id'][n + 1])
            class_ids.append(data_dict['class_id'][m])
            class_ids.append(data_dict['class_id'][m2])
                

        # indexes = np.arange(len(mixtures))  # (0, 1, ..., batch_size)
        # if with_identity_zero:
        #     indexes = np.arange(len(mixtures))  # (0, 1, ..., batch_size)
        # else:
        #     N = len(mixtures)
        #     indexes = np.arange(N).reshape((N // 4, 4))[:, 0 : 2].flatten()
        #     """(0, 1, 4, 5, 8, 9, ..., batch_size * 2 - 4, batch_size * 2 - 3)"""

        output_dict = {
            'class_id': np.array(class_ids),  # (batch_size,)
            'mixture': np.array(mixtures)[:, :, None], # (batch_size,) | (batch_size * 2,)
            'source': np.array(sources)[:, :, None],   # (batch_size,) | (batch_size * 2,)
            'hard_condition': np.array(hard_conditions),   # (batch_size,) | (batch_size * 2,)
            }

        return output_dict

    def get_mix_data3(self, data_dict): 

        framewise_output = self.sed_model.inference(data_dict['waveform'])
        (audios_num, total_frames_num, classes_num) = framewise_output.shape

        seg_waveforms = []
        for n in range(audios_num):
            smoothed_framewise_output = np.convolve(
                framewise_output[n, :, data_dict['class_id'][n]], np.ones(self.segment_frames), mode='same')
            anchor_index = np.argmax(smoothed_framewise_output)

            (bgn_sample, end_sample) = self.get_segment_bgn_end_samples(
                anchor_index, self.segment_frames)

            seg_waveforms.append(data_dict['waveform'][n, bgn_sample : end_sample])

        seg_waveforms = np.array(seg_waveforms)
        seg_predictions, _ = self.at_model.inference(seg_waveforms)
        
        mixtures = []
        sources = []
        soft_conditions = []
        hard_conditions = []
        class_ids = []
        for n in range(0, audios_num, 2):
            ratio = (calculate_average_energy(seg_waveforms[n]) / max(1e-8, calculate_average_energy(seg_waveforms[n + 1]))) ** 0.5
            ratio = np.clip(ratio, 0.02, 50)
            seg_waveforms[n + 1] *= ratio
            mixture = seg_waveforms[n] + seg_waveforms[n + 1]

            if False:
                import crash
                asdf 
                K = 3
                config.ix_to_lb[data_dict['class_id'][K]]
                data_dict['target'][K]
                [config.ix_to_lb[idx] for idx in np.argsort(seg_predictions[K])[::-1][0:10]]
                librosa.output.write_wav('_zz.wav', data_dict['waveform'][K], sr=32000)
                librosa.output.write_wav('_zz2.wav', seg_waveforms[K], sr=32000)
                seg_predictions[K, data_dict['class_id'][K]]
                np.max(framewise_output[K, :, data_dict['class_id'][K]])

            # Mixutres
            mixtures.append(mixture)
            mixtures.append(mixture)
            _rnd = self.random_state.randint(2)
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

            class_ids.append(data_dict['class_id'][n])
            class_ids.append(data_dict['class_id'][n + 1])
            class_ids.append(data_dict['class_id'][m])
            class_ids.append(data_dict['class_id'][m2])
                
        output_dict = {
            'class_id': np.array(class_ids),  # (batch_size,)
            'mixture': np.array(mixtures)[:, :, None], # (batch_size,) | (batch_size * 2,)
            'source': np.array(sources)[:, :, None],   # (batch_size,) | (batch_size * 2,)
            'hard_condition': np.array(hard_conditions),   # (batch_size,) | (batch_size * 2,)
            'soft_condition': np.array(soft_conditions)
            }

        return output_dict

    def get_mix_data4(self, data_dict): 

        framewise_output = self.sed_model.inference(data_dict['waveform'])
        (audios_num, total_frames_num, classes_num) = framewise_output.shape

        seg_waveforms = []
        for n in range(audios_num):
            smoothed_framewise_output = np.convolve(
                framewise_output[n, :, data_dict['class_id'][n]], np.ones(self.segment_frames), mode='same')
            anchor_index = np.argmax(smoothed_framewise_output)

            (bgn_sample, end_sample) = self.get_segment_bgn_end_samples(
                anchor_index, self.segment_frames)

            seg_waveforms.append(data_dict['waveform'][n, bgn_sample : end_sample])

        seg_waveforms = np.array(seg_waveforms)
        seg_predictions, _ = self.at_model.inference(seg_waveforms)
 
        pred_ids = []
        for i in range(seg_predictions.shape[0]):
            tmp = []
            for j in range(seg_predictions.shape[1]):
                if seg_predictions[i, j] > self.opt_thres[j] / 2:
                    tmp.append(j)
            tmp.append(data_dict['class_id'][i])
            tmp = list(set(tmp))
            pred_ids.append(tmp)

        mixtures = []
        sources = []
        soft_conditions = []
        hard_conditions = []
        class_ids = []
        for n in range(0, audios_num, 2):
            ratio = (calculate_average_energy(seg_waveforms[n]) / max(1e-8, calculate_average_energy(seg_waveforms[n + 1]))) ** 0.5
            ratio = np.clip(ratio, 0.02, 50)
            seg_waveforms[n + 1] *= ratio
            mixture = seg_waveforms[n] + seg_waveforms[n + 1]

            if False:
                K = 3
                config.ix_to_lb[data_dict['class_id'][K]]
                data_dict['target'][K]
                [config.ix_to_lb[idx] for idx in np.argsort(seg_predictions[K])[::-1][0:10]]
                librosa.output.write_wav('_zz.wav', data_dict['waveform'][K], sr=32000)
                librosa.output.write_wav('_zz2.wav', seg_waveforms[K], sr=32000)
                seg_predictions[K, data_dict['class_id'][K]]
                np.max(framewise_output[K, :, data_dict['class_id'][K]])

                tmp = []
                for j in range(527):
                    if seg_predictions[K, j] > self.opt_thres[j] / 2:
                        tmp.append(j)

                import crash
                asdf

            # Mixutres
            mixtures.append(mixture)
            mixtures.append(mixture)
            _rnd = self.random_state.randint(2)
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
            hard_conditions.append(ids_to_hots(pred_ids[n], classes_num))
            hard_conditions.append(ids_to_hots(pred_ids[n + 1], classes_num))
            hard_conditions.append(ids_to_hots(pred_ids[m], classes_num))
            hard_conditions.append(ids_to_hots(pred_ids[m2], classes_num))

            class_ids.append(data_dict['class_id'][n])
            class_ids.append(data_dict['class_id'][n + 1])
            class_ids.append(data_dict['class_id'][m])
            class_ids.append(data_dict['class_id'][m2])
                
        output_dict = {
            'class_id': np.array(class_ids),  # (batch_size,)
            'mixture': np.array(mixtures)[:, :, None], # (batch_size,) | (batch_size * 2,)
            'source': np.array(sources)[:, :, None],   # (batch_size,) | (batch_size * 2,)
            'hard_condition': np.array(hard_conditions),   # (batch_size,) | (batch_size * 2,)
            'soft_condition': np.array(soft_conditions)
            }

        return output_dict

    def get_mix_data4b(self, data_dict): 

        framewise_output = self.sed_model.inference(data_dict['waveform'])
        (audios_num, total_frames_num, classes_num) = framewise_output.shape

        seg_waveforms = []
        for n in range(audios_num):
            smoothed_framewise_output = np.convolve(
                framewise_output[n, :, data_dict['class_id'][n]], np.ones(self.segment_frames), mode='same')
            anchor_index = np.argmax(smoothed_framewise_output)

            (bgn_sample, end_sample) = self.get_segment_bgn_end_samples(
                anchor_index, self.segment_frames)

            seg_waveforms.append(data_dict['waveform'][n, bgn_sample : end_sample])

        seg_waveforms = np.array(seg_waveforms)
        seg_predictions, _ = self.at_model.inference(seg_waveforms)
 
        pred_ids = []
        for i in range(seg_predictions.shape[0]):
            tmp = []
            for j in range(seg_predictions.shape[1]):
                if seg_predictions[i, j] > self.opt_thres[j] / 2:
                    tmp.append(j)
            tmp.append(data_dict['class_id'][i])
            tmp = list(set(tmp))
            pred_ids.append(tmp)

        mixtures = []
        sources = []
        soft_conditions = []
        hard_conditions = []
        class_ids = []
        for n in range(0, audios_num, 2):
            ratio = (calculate_average_energy(seg_waveforms[n]) / max(1e-8, calculate_average_energy(seg_waveforms[n + 1]))) ** 0.5
            ratio = np.clip(ratio, 0.02, 50)
            seg_waveforms[n + 1] *= ratio
            mixture = seg_waveforms[n] + seg_waveforms[n + 1]

            if False:
                K = 3
                config.ix_to_lb[data_dict['class_id'][K]]
                data_dict['target'][K]
                [config.ix_to_lb[idx] for idx in np.argsort(seg_predictions[K])[::-1][0:10]]
                librosa.output.write_wav('_zz.wav', data_dict['waveform'][K], sr=32000)
                librosa.output.write_wav('_zz2.wav', seg_waveforms[K], sr=32000)
                seg_predictions[K, data_dict['class_id'][K]]
                np.max(framewise_output[K, :, data_dict['class_id'][K]])

                tmp = []
                for j in range(527):
                    if seg_predictions[K, j] > self.opt_thres[j] / 2:
                        tmp.append(j)

                import crash
                asdf

            # Mixutres
            mixtures.append(mixture)
            mixtures.append(mixture)
            _rnd = self.random_state.randint(2)
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

            for j, id in enumerate(pred_ids[m2]):
                if id in pred_ids[m]:
                    pred_ids[m2][j] = 0

            # Hard conditions
            hard_conditions.append(ids_to_hots(pred_ids[n], classes_num))
            hard_conditions.append(ids_to_hots(pred_ids[n + 1], classes_num))
            hard_conditions.append(ids_to_hots(pred_ids[m], classes_num))
            hard_conditions.append(ids_to_hots(pred_ids[m2], classes_num))

            class_ids.append(data_dict['class_id'][n])
            class_ids.append(data_dict['class_id'][n + 1])
            class_ids.append(data_dict['class_id'][m])
            class_ids.append(data_dict['class_id'][m2])
                
        output_dict = {
            'class_id': np.array(class_ids),  # (batch_size,)
            'mixture': np.array(mixtures)[:, :, None], # (batch_size,) | (batch_size * 2,)
            'source': np.array(sources)[:, :, None],   # (batch_size,) | (batch_size * 2,)
            'hard_condition': np.array(hard_conditions),   # (batch_size,) | (batch_size * 2,)
            'soft_condition': np.array(soft_conditions)
            }

        return output_dict

    def get_mix_data5(self, data_dict): 

        def _add(x):
            indexes = [*range(x.shape[0])]
            new_indexes = []

            while indexes:
                i = indexes[0]
                indexes.remove(i)
                for j in indexes:
                    if np.sum(x[i] * x[j]) == 0:
                        new_indexes.append(i)
                        new_indexes.append(j)
                        indexes.remove(j)
                        break
            
            return new_indexes

        framewise_output = self.sed_model.inference(data_dict['waveform'])
        (audios_num, total_frames_num, classes_num) = framewise_output.shape

        seg_waveforms = []
        for n in range(audios_num):
            smoothed_framewise_output = np.convolve(
                framewise_output[n, :, data_dict['class_id'][n]], np.ones(self.segment_frames), mode='same')
            anchor_index = np.argmax(smoothed_framewise_output)

            (bgn_sample, end_sample) = self.get_segment_bgn_end_samples(
                anchor_index, self.segment_frames)

            seg_waveforms.append(data_dict['waveform'][n, bgn_sample : end_sample])

        seg_waveforms = np.array(seg_waveforms)
        seg_predictions, _ = self.at_model.inference(seg_waveforms)

        tmp = np.zeros_like(seg_predictions)
        for i in range(seg_predictions.shape[0]):
            for j in range(seg_predictions.shape[1]):
                if seg_predictions[i, j] > self.opt_thres[j] / 2:
                    tmp[i, j] = 1

        indexes = _add(tmp)

        mixtures = []
        sources = []
        soft_conditions = []
        hard_conditions = []
        class_ids = []
        for i in range(0, len(indexes), 2):
            n1 = indexes[i]
            n2 = indexes[i + 1]
            ratio = (calculate_average_energy(seg_waveforms[n1]) / max(1e-8, calculate_average_energy(seg_waveforms[n2]))) ** 0.5
            ratio = np.clip(ratio, 0.02, 50)
            seg_waveforms[n2] *= ratio
            mixture = seg_waveforms[n1] + seg_waveforms[n2]

            if False:
                import crash
                asdf 
                K = 10
                config.ix_to_lb[data_dict['class_id'][K]]
                data_dict['target'][K]
                [config.ix_to_lb[idx] for idx in np.argsort(seg_predictions[K])[::-1][0:10]]
                librosa.output.write_wav('_zz.wav', data_dict['waveform'][K], sr=32000)
                librosa.output.write_wav('_zz2.wav', seg_waveforms[K], sr=32000)
                seg_predictions[K, data_dict['class_id'][K]]
                # seg_predictions[K]
                # np.max(framewise_output[K, :, data_dict['class_id'][K]])

            # Mixutres
            mixtures.append(mixture)
            mixtures.append(mixture)
            _rnd = self.random_state.randint(2)
            if _rnd == 0:
                m1 = n1
                m2 = n2
            elif _rnd == 1:
                m1 = n2
                m2 = n1
            
            mixtures.append(seg_waveforms[m1])
            mixtures.append(seg_waveforms[m1])

            # Targets
            sources.append(seg_waveforms[n1])
            sources.append(seg_waveforms[n2])
            sources.append(seg_waveforms[m1])
            sources.append(np.zeros_like(seg_waveforms[m1]))
            
            # Soft conditions
            soft_conditions.append(seg_predictions[n1])
            soft_conditions.append(seg_predictions[n2])
            soft_conditions.append(seg_predictions[m1])

            # f(x1, c2) -> 0. Make sure c2 and the prediction of x1 is exclusive. 
            for k in range(classes_num):
                if seg_predictions[m1, k] >= 0.02:
                    seg_predictions[m2, k] = 0.
            soft_conditions.append(seg_predictions[m2])
            
            # Hard conditions
            hard_conditions.append(id_to_one_hot(data_dict['class_id'][n1], classes_num))
            hard_conditions.append(id_to_one_hot(data_dict['class_id'][n2], classes_num))
            hard_conditions.append(id_to_one_hot(data_dict['class_id'][m1], classes_num))
            hard_conditions.append(id_to_one_hot(data_dict['class_id'][m2], classes_num))

            class_ids.append(data_dict['class_id'][n1])
            class_ids.append(data_dict['class_id'][n2])
            class_ids.append(data_dict['class_id'][m1])
            class_ids.append(data_dict['class_id'][m2])
                
        output_dict = {
            'class_id': np.array(class_ids),  # (batch_size,)
            'mixture': np.array(mixtures)[:, :, None], # (batch_size,) | (batch_size * 2,)
            'source': np.array(sources)[:, :, None],   # (batch_size,) | (batch_size * 2,)
            'hard_condition': np.array(hard_conditions),   # (batch_size,) | (batch_size * 2,)
            'soft_condition': np.array(soft_conditions)
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

