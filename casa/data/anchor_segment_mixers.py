import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorSegmentMixer(nn.Module):
    def __init__(self, mix_num):
        super(AnchorSegmentMixer, self).__init__()

        self.mix_num = mix_num

    def __call__(self, waveforms):
        
        # waveforms = segments_dict['waveform']

        batch_size = waveforms.shape[0]

        data_dict = {
            'segment': [],
            'mixture': [],
        }

        for n in range(0, batch_size):

            segment = waveforms[n].clone()
            mixture = waveforms[n].clone()

            for i in range(1, self.mix_num):
                next_segment = waveforms[(n + i) % batch_size]
                rescaled_next_segment = rescale_to_match_energy(next_segment, segment)
                mixture += rescaled_next_segment

            data_dict['segment'].append(segment)
            data_dict['mixture'].append(mixture)

        for key in data_dict.keys():
            data_dict[key] = torch.stack(data_dict[key], dim=0)

        # return data_dict
        return data_dict['mixture'], data_dict['segment']


def rescale_to_match_energy(segment1, segment2):

    ratio = get_energy_ratio(segment1, segment2)
    recaled_segment1 = segment1 / ratio
    return recaled_segment1


def get_energy(x):
    return torch.mean(x ** 2)


def get_energy_ratio(segment1, segment2):

    energy1 = get_energy(segment1)
    energy2 = max(get_energy(segment2), 1e-10)
    ratio = (energy1 / energy2) ** 0.5
    ratio = torch.clamp(ratio, 0.02, 50)
    return ratio