from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorSegmentMixer(nn.Module):
    def __init__(self, mix_num: int) -> None:
        r"""Anchor segment mixer. Used to mix multiple sources into a mixture.

        Args:
            mix_num (int): the number of sources to mix

        Returns:
            None
        """

        super(AnchorSegmentMixer, self).__init__()

        self.mix_num = mix_num

    def __call__(self, waveforms: torch.Tensor) -> Dict:
        r"""Mix multiple sources to mixture.

        Args:
            waveforms (torch.Tensor): (batch_size, segment_samples)

        Returns:
            mixtures (torch.Tensor): (batch_size, segment_samples)
            targets (torch.Tensor): (batch_size, segment_samples)
        """

        batch_size = waveforms.shape[0]

        targets = []
        mixtures = []

        for n in range(0, batch_size):

            segment = waveforms[n].clone()
            mixture = waveforms[n].clone()

            for i in range(1, self.mix_num):

                next_segment = waveforms[(n + i) % batch_size]

                # Rescale the energy of the next_segment to match the energy of 
                # the segment
                rescaled_next_segment = rescale_to_match_energy(segment, next_segment)
                
                mixture += rescaled_next_segment

            targets.append(segment)
            mixtures.append(mixture)

        targets = torch.stack(targets, dim=0)
        mixtures = torch.stack(mixtures, dim=0)
        
        return mixtures, targets


def rescale_to_match_energy(
    segment1: torch.Tensor, 
    segment2: torch.Tensor,
) -> torch.Tensor:
    r"""Rescale segment2 to match the energy of segment1."""

    ratio = get_energy_ratio(segment1, segment2)
    recaled_segment2 = segment2 * ratio
    return recaled_segment2


def get_energy(x: torch.Tensor) -> torch.float:
    r"""Calculate the energy of a signal."""

    return torch.mean(x ** 2)


def get_energy_ratio(
    segment1: torch.Tensor, 
    segment2: torch.Tensor, 
    eps=1e-10
) -> float:
    r"""Calculate ratio = sqrt(E(s1) / E(s2))."""

    energy1 = get_energy(segment1)
    energy2 = get_energy(segment2)
    ratio = (energy1 / max(energy2, eps)) ** 0.5
    ratio = torch.clamp(ratio, 0.02, 50)

    return ratio