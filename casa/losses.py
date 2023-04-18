import torch
import torch.nn as nn

from casa.models.base import Base
from torchlibrosa.stft import STFT


def l1(output, target):
    return torch.mean(torch.abs(output - target))


def l1_wav(output_dict, target_dict):
	return l1(output_dict['segment'], target_dict['segment'])


class L1_Wav_L1_Sp(nn.Module, Base):
    def __init__(self):
        r"""L1 loss in the time-domain and L1 loss on the spectrogram."""
        super(L1_Wav_L1_Sp, self).__init__()

        self.window_size = 2048
        hop_size = 320
        center = True
        pad_mode = "reflect"
        window = "hann"

        self.stft = STFT(
            n_fft=self.window_size,
            hop_length=hop_size,
            win_length=self.window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

    def __call__(
        self, output_dict, target_dict) -> torch.Tensor:
        r"""L1 loss in the time-domain and on the spectrogram.

        Args:
            output: torch.Tensor
            target: torch.Tensor

        Returns:
            loss: torch.float
        """

        # L1 loss in the time-domain.
        wav_loss = l1_wav(output_dict, target_dict)
        
        # L1 loss on the spectrogram.
        sp_loss = l1(
            self.wav_to_spectrogram(output_dict["segment"], eps=1e-8),
            self.wav_to_spectrogram(target_dict["segment"], eps=1e-8),
        )

        # Total loss.
        return wav_loss + sp_loss



def get_loss_function(loss_type):
    if loss_type == "l1_wav":
        return l1_wav

    elif loss_type == "l1_wav_l1_sp":
    	return L1_Wav_L1_Sp()

    else:
        raise NotImplementedError("Error!")
