import numpy as np
import librosa
import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.parameter import Parameter
from pytorch_utils import move_data_to_device

from torchlibrosa.stft import STFT, ISTFT, magphase


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_emb(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.uniform_(layer.weight, -0.1, 0.1)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


def act(x, activation):
    if activation == 'relu':
        return F.relu_(x)

    elif activation == 'leaky_relu':
        return F.leaky_relu_(x, negative_slope=0.2)

    elif activation == 'swish':
        return x * torch.sigmoid(x)

    else:
        raise Exception('Incorrect activation!')


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size, activation, momentum):
        super(ConvBlock, self).__init__()

        self.activation = activation
        pad = size // 2
        classes_num = 527

        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(size, size), stride=(1, 1), 
                              dilation=(1, 1), padding=(pad, pad), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(size, size), stride=(1, 1), 
                              dilation=(1, 1), padding=(pad, pad), bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.emb1 = nn.Linear(classes_num, out_channels, bias=True)
        self.emb2 = nn.Linear(classes_num, out_channels, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_emb(self.emb1)
        init_emb(self.emb2)

    def forward(self, x, condition):
        c1 = self.emb1(condition)
        c2 = self.emb2(condition)
        x = act(self.bn1(self.conv1(x)), self.activation) + c1[:, :, None, None]
        x = act(self.bn2(self.conv2(x)), self.activation) + c2[:, :, None, None]
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, activation, momentum):
        super(EncoderBlock, self).__init__()
        size = 3

        self.conv_block = ConvBlock(in_channels, out_channels, size, activation, momentum)
        self.downsample = downsample

    def forward(self, x, condition):
        encoder = self.conv_block(x, condition)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation, momentum):
        super(DecoderBlock, self).__init__()
        size = 3
        self.activation = activation
        classes_num = 527

        self.conv1 = torch.nn.ConvTranspose2d(in_channels=in_channels, 
            out_channels=out_channels, kernel_size=(size, size), stride=stride, 
            padding=(0, 0), output_padding=(0, 0), bias=False, dilation=(1, 1))

        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.conv_block2 = ConvBlock(out_channels * 2, out_channels, size, activation, momentum)

        self.emb1 = nn.Linear(classes_num, out_channels, bias=True)

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn)
        init_emb(self.emb1)

    def prune(self, x):
        """Prune the shape of x after transpose convolution.
        """
        x = x[:, :, 0 : - 1, 0 : - 1]
        return x

    def forward(self, input_tensor, concat_tensor, condition):
        c1 = self.emb1(condition)
        x = act(self.bn1(self.conv1(input_tensor)), self.activation) + c1[:, :, None, None]
        x = self.prune(x)
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x, condition)
        return x


class UNet(nn.Module):
    def __init__(self, channels):
        super(UNet, self).__init__()

        window_size = 2048
        hop_size = 441
        center = True
        pad_mode = 'reflect'
        window = 'hann'
        activation = 'relu'
        momentum = 0.01

        self.downsample_ratio = 2 ** 6   # This number equals 2^{#encoder_blcoks}

        self.stft = STFT(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, 
            pad_mode=pad_mode, freeze_parameters=True)

        self.istft = ISTFT(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, 
            pad_mode=pad_mode, freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)

        self.encoder_block1 = EncoderBlock(in_channels=channels, out_channels=32, 
            downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block2 = EncoderBlock(in_channels=32, out_channels=64, 
            downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block3 = EncoderBlock(in_channels=64, out_channels=128, 
            downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block4 = EncoderBlock(in_channels=128, out_channels=256, 
            downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block5 = EncoderBlock(in_channels=256, out_channels=512, 
            downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block6 = EncoderBlock(in_channels=512, out_channels=1024, 
            downsample=(2, 2), activation=activation, momentum=momentum)
        self.conv_block7 = ConvBlock(in_channels=1024, out_channels=2048, 
            size=3, activation=activation, momentum=momentum)
        self.decoder_block1 = DecoderBlock(in_channels=2048, out_channels=1024, 
            stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block2 = DecoderBlock(in_channels=1024, out_channels=512, 
            stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block3 = DecoderBlock(in_channels=512, out_channels=256, 
            stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block4 = DecoderBlock(in_channels=256, out_channels=128, 
            stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block5 = DecoderBlock(in_channels=128, out_channels=64, 
            stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block6 = DecoderBlock(in_channels=64, out_channels=32, 
            stride=(2, 2), activation=activation, momentum=momentum)

        self.after_conv_block1 = ConvBlock(in_channels=32, out_channels=32, 
            size=3, activation=activation, momentum=momentum)

        self.after_conv2 = nn.Conv2d(in_channels=32, out_channels=channels, 
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.after_conv2)

    def spectrogram(self, input):
        (real, imag) = self.stft(input)
        return (real ** 2 + imag ** 2) ** 0.5

    def wav_to_spectrogram(self, input):
        """Waveform to spectrogram.

        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        channels_num = input.shape[2]
        for channel in range(channels_num):
            sp_list.append(self.spectrogram(input[:, :, channel]))

        output = torch.cat(sp_list, dim=1)
        return output


    def spectrogram_to_wav(self, input, spectrogram, length=None):
        """Spectrogram to waveform.

        Args:
          input: (batch_size, segment_samples, channels_num)
          spectrogram: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, segment_samples, channels_num)
        """
        channels_num = input.shape[2]
        wav_list = []
        for channel in range(channels_num):
            (real, imag) = self.stft(input[:, :, channel])
            (_, cos, sin) = magphase(real, imag)
            wav_list.append(self.istft(spectrogram[:, channel : channel + 1, :, :] * cos, 
                spectrogram[:, channel : channel + 1, :, :] * sin, length))
        
        output = torch.stack(wav_list, dim=2)
        return output

    def forward(self, input, condition):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        sp = self.wav_to_spectrogram(input)    
        """(batch_size, channels_num, time_steps, freq_bins)"""

        # Batch normalization
        x = sp.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = int(np.ceil(x.shape[2] / self.downsample_ratio)) \
            * self.downsample_ratio - origin_len
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0 : x.shape[-1] - 1]     # (bs, channels, T, F)

        # UNet
        (x1_pool, x1) = self.encoder_block1(x, condition)  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool, condition)    # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool, condition)    # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(x3_pool, condition)    # x4_pool: (bs, 256, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(x4_pool, condition)    # x5_pool: (bs, 512, T / 32, F / 32)
        (x6_pool, x6) = self.encoder_block6(x5_pool, condition)    # x6_pool: (bs, 1024, T / 64, F / 64)
        x_center = self.conv_block7(x6_pool, condition)    # (bs, 2048, T / 64, F / 64)
        x7 = self.decoder_block1(x_center, x6, condition)  # (bs, 1024, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5, condition)    # (bs, 512, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4, condition)    # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3, condition)   # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2, condition)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1, condition)  # (bs, 32, T, F)
        x = self.after_conv_block1(x12, condition)     # (bs, 32, T, F)
        x = self.after_conv2(x)             # (bs, channels, T, F)

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0 : origin_len, :]

        sp_out = torch.sigmoid(x) * sp

        # Spectrogram to wav
        length = input.shape[1]
        wav_out = self.spectrogram_to_wav(input, sp_out, length)

        output_dict = {'wav': wav_out, 'sp': sp_out}
        return output_dict


class UNet_16k(nn.Module):
    def __init__(self, channels):
        super(UNet_16k, self).__init__()

        window_size = 1024
        hop_size = 160
        center = True
        pad_mode = 'reflect'
        window = 'hann'
        activation = 'relu'
        momentum = 0.01

        self.downsample_ratio = 2 ** 6   # This number equals 2^{#encoder_blcoks}

        self.stft = STFT(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, 
            pad_mode=pad_mode, freeze_parameters=True)

        self.istft = ISTFT(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, 
            pad_mode=pad_mode, freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)

        self.encoder_block1 = EncoderBlock(in_channels=channels, out_channels=32, 
            downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block2 = EncoderBlock(in_channels=32, out_channels=64, 
            downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block3 = EncoderBlock(in_channels=64, out_channels=128, 
            downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block4 = EncoderBlock(in_channels=128, out_channels=256, 
            downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block5 = EncoderBlock(in_channels=256, out_channels=512, 
            downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block6 = EncoderBlock(in_channels=512, out_channels=1024, 
            downsample=(2, 2), activation=activation, momentum=momentum)
        self.conv_block7 = ConvBlock(in_channels=1024, out_channels=2048, 
            size=3, activation=activation, momentum=momentum)
        self.decoder_block1 = DecoderBlock(in_channels=2048, out_channels=1024, 
            stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block2 = DecoderBlock(in_channels=1024, out_channels=512, 
            stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block3 = DecoderBlock(in_channels=512, out_channels=256, 
            stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block4 = DecoderBlock(in_channels=256, out_channels=128, 
            stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block5 = DecoderBlock(in_channels=128, out_channels=64, 
            stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block6 = DecoderBlock(in_channels=64, out_channels=32, 
            stride=(2, 2), activation=activation, momentum=momentum)

        self.after_conv_block1 = ConvBlock(in_channels=32, out_channels=32, 
            size=3, activation=activation, momentum=momentum)

        self.after_conv2 = nn.Conv2d(in_channels=32, out_channels=channels, 
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.after_conv2)

    def spectrogram(self, input):
        (real, imag) = self.stft(input)
        return (real ** 2 + imag ** 2) ** 0.5

    def wav_to_spectrogram(self, input):
        """Waveform to spectrogram.

        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        channels_num = input.shape[2]
        for channel in range(channels_num):
            sp_list.append(self.spectrogram(input[:, :, channel]))

        output = torch.cat(sp_list, dim=1)
        return output


    def spectrogram_to_wav(self, input, spectrogram, length=None):
        """Spectrogram to waveform.

        Args:
          input: (batch_size, segment_samples, channels_num)
          spectrogram: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, segment_samples, channels_num)
        """
        channels_num = input.shape[2]
        wav_list = []
        for channel in range(channels_num):
            (real, imag) = self.stft(input[:, :, channel])
            (_, cos, sin) = magphase(real, imag)
            wav_list.append(self.istft(spectrogram[:, channel : channel + 1, :, :] * cos, 
                spectrogram[:, channel : channel + 1, :, :] * sin, length))
        
        output = torch.stack(wav_list, dim=2)
        return output

    def forward(self, input, condition):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        sp = self.wav_to_spectrogram(input)    
        """(batch_size, channels_num, time_steps, freq_bins)"""

        # Batch normalization
        x = sp.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = int(np.ceil(x.shape[2] / self.downsample_ratio)) \
            * self.downsample_ratio - origin_len
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0 : x.shape[-1] - 1]     # (bs, channels, T, F)

        # UNet
        (x1_pool, x1) = self.encoder_block1(x, condition)  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool, condition)    # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool, condition)    # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(x3_pool, condition)    # x4_pool: (bs, 256, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(x4_pool, condition)    # x5_pool: (bs, 512, T / 32, F / 32)
        (x6_pool, x6) = self.encoder_block6(x5_pool, condition)    # x6_pool: (bs, 1024, T / 64, F / 64)
        x_center = self.conv_block7(x6_pool, condition)    # (bs, 2048, T / 64, F / 64)
        x7 = self.decoder_block1(x_center, x6, condition)  # (bs, 1024, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5, condition)    # (bs, 512, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4, condition)    # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3, condition)   # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2, condition)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1, condition)  # (bs, 32, T, F)
        x = self.after_conv_block1(x12, condition)     # (bs, 32, T, F)
        x = self.after_conv2(x)             # (bs, channels, T, F)

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0 : origin_len, :]

        sp_out = torch.sigmoid(x) * sp

        # Spectrogram to wav
        length = input.shape[1]
        wav_out = self.spectrogram_to_wav(input, sp_out, length)

        output_dict = {'wav': wav_out, 'sp': sp_out}

        return output_dict