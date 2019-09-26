import torch
import torch.nn.functional as F


def mae(output_stft, vocal_stft):
    """Mean absolute error."""
    return F.l1_loss(output_stft, vocal_stft)


def get_loss_func(loss_type):
    if loss_type == 'mae':
        return mae