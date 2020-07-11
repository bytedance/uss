import torch.nn.functional as F
import torch
import numpy as np


def mae(input, target):
    return torch.mean(torch.abs(input - target))


def logmae_wav(model, output_dict, target):
    loss = torch.log10(torch.clamp(mae(output_dict['wav'], target), 1e-8, np.inf))
    return loss


def get_loss_func(loss_type):
    if loss_type == 'logmae_wav':
        return logmae_wav

    elif loss_type == 'mae':
    	return mae

    else:
        raise Exception('Incorrect loss_type!')
