import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from stft import STFT, ISTFT, magphase


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_channels, out_channels, classes_num, kernel_size, condition_type):
        super(double_conv, self).__init__()
        
        self.condition_type = condition_type
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if condition_type == 'soft':
            cond_nums = classes_num
        elif condition_type == 'soft_hard':
            cond_nums = classes_num * 2
        else:
            raise Exception('Incorrect condition_type!')

        self.fc_cond1 = nn.Linear(cond_nums, out_channels, bias=True)
        self.fc_cond2 = nn.Linear(cond_nums, out_channels, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        init_layer(self.fc_cond1)
        init_layer(self.fc_cond2)
        
    def forward(self, x, hard_cond, soft_cond):
        if self.condition_type == 'soft':
            condition = soft_cond
        elif self.condition_type == 'soft_hard':
            condition = torch.cat((hard_cond, soft_cond), dim=-1)

        cond1 = self.fc_cond1(condition)
        cond2 = self.fc_cond2(condition)
        x = F.relu_(self.bn1(self.conv1(x)) + cond1[:, :, None, None])
        x = F.relu_(self.bn2(self.conv2(x)) + cond2[:, :, None, None])
        return x


class inconv(nn.Module):
    def __init__(self, in_channels, out_channels, classes_num, kernel_size, condition_type):
        super(inconv, self).__init__()
        self.conv = double_conv(in_channels, out_channels, classes_num, kernel_size, condition_type)

    def forward(self, x, hard_cond, soft_cond):
        x = self.conv(x, hard_cond, soft_cond)
        return x


class down(nn.Module):
    def __init__(self, in_channels, out_channels, classes_num, kernel_size, condition_type):
        super(down, self).__init__()
        self.conv_block = double_conv(in_channels, out_channels, classes_num, kernel_size, condition_type)

    def forward(self, x, hard_cond, soft_cond):
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = self.conv_block(x, hard_cond, soft_cond)
        return x


class up(nn.Module):
    def __init__(self, in_channels, out_channels, classes_num, kernel_size, condition_type):
        super(up, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)
        self.conv = double_conv(in_channels, out_channels, classes_num, kernel_size, condition_type)

        self.init_weights()

    def init_weights(self):
        init_layer(self.up)

    def forward(self, x1, x2, hard_cond, soft_cond):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x, hard_cond, soft_cond)
        return x


class outconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, classes_num, condition_type, wiener_filter):
        super(UNet, self).__init__()
 
        window_size = 1024
        hop_size = 256
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        device = 'cuda'
        self.wiener_filter = wiener_filter

        self.stft = STFT(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        self.istft = ISTFT(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1)

        kernel_size = (3, 3)
        self.inc = inconv(1, 64, classes_num, kernel_size, condition_type)
        self.down1 = down(64, 128, classes_num, kernel_size, condition_type)
        self.down2 = down(128, 256, classes_num, kernel_size, condition_type)
        self.down3 = down(256, 512, classes_num, kernel_size, condition_type)
        self.down4 = down(512, 512, classes_num, kernel_size, condition_type)
        self.up1 = up(1024, 256, classes_num, kernel_size, condition_type)
        self.up2 = up(512, 128, classes_num, kernel_size, condition_type)
        self.up3 = up(256, 64, classes_num, kernel_size, condition_type)
        self.up4 = up(128, 64, classes_num, kernel_size, condition_type)
        self.outc = outconv(64, 1)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        
    def spectrogram(self, input):
        (real, imag) = self.stft(input)
        return (real ** 2 + imag ** 2) ** 0.5

    def wavin_to_specin(self, input):
        return self.spectrogram(input)

    def wavin_to_target(self, input):
        return self.spectrogram(input)

    def specout_to_stft_magnitude(self, input):
        return input

    def wavin_to_wavout(self, input, hard_cond, soft_cond, length=None):

        specout = self.forward(input, hard_cond, soft_cond)
        
        stft_magnitude = self.specout_to_stft_magnitude(specout)

        (real, imag) = self.stft(input)
        (_, cos, sin) = magphase(real, imag)
        wavout = self.istft(stft_magnitude * cos, stft_magnitude * sin, length)

        return wavout

    def forward(self, input, hard_cond, soft_cond):

        sp = self.wavin_to_specin(input)
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = sp.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x1 = self.inc(x, hard_cond, soft_cond)
        x2 = self.down1(x1, hard_cond, soft_cond)
        x3 = self.down2(x2, hard_cond, soft_cond)
        x4 = self.down3(x3, hard_cond, soft_cond)
        x5 = self.down4(x4, hard_cond, soft_cond)
        x = self.up1(x5, x4, hard_cond, soft_cond)
        x = self.up2(x, x3, hard_cond, soft_cond)
        x = self.up3(x, x2, hard_cond, soft_cond)
        x = self.up4(x, x1, hard_cond, soft_cond)
        x = self.outc(x)
        
        if self.wiener_filter:
            out = torch.sigmoid(x) * sp
        else:
            out = F.relu_(x)

        return out