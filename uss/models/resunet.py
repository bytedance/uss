from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchlibrosa.stft import ISTFT, STFT, magphase

from uss.models.base import Base, init_bn, init_layer
from uss.models.film import FiLM, get_film_meta


class ConvBlockRes(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        momentum: float,
        has_film: bool,
    ) -> None:
        r"""Residual convolutional block."""

        super(ConvBlockRes, self).__init__()

        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.has_film = has_film

        self.init_weights()

    def init_weights(self) -> None:
        r"""Initialize weights."""

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self,
                input_tensor: torch.Tensor,
                film_dict: Dict
                ) -> torch.Tensor:
        r"""Forward input feature maps to the encoder block.

        Args:
            input_tensor (torch.Tensor), (B, C, T, F)
            film_dict (Dict)

        Returns:
            output (torch.Tensor), (B, C, T, F)
        """

        b1 = film_dict['beta1']
        b2 = film_dict['beta2']

        x = self.conv1(
            F.leaky_relu_(
                self.bn1(input_tensor) + b1,
                negative_slope=0.01))
        x = self.conv2(F.leaky_relu_(self.bn2(x) + b2, negative_slope=0.01))

        if self.is_shortcut:
            output = self.shortcut(input_tensor) + x
        else:
            output = input_tensor + x

        return output


class EncoderBlockRes1B(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        downsample: Tuple,
        momentum: float,
        has_film: bool,
    ) -> None:
        r"""Encoder block."""

        super(EncoderBlockRes1B, self).__init__()

        self.conv_block1 = ConvBlockRes(
            in_channels, out_channels, kernel_size, momentum, has_film,
        )
        self.downsample = downsample

    def forward(self,
                input_tensor: torch.Tensor,
                film_dict: Dict
                ) -> torch.Tensor:
        r"""Forward input feature maps to the encoder block.

        Args:
            input_tensor (torch.Tensor), (B, C_in, T, F)
            film_dict (Dict)

        Returns:
            encoder (torch.Tensor): (B, C_out, T, F)
            encoder_pool (torch.Tensor): (B, C_out, T / downsample, F / downsample)
        """

        encoder = self.conv_block1(input_tensor, film_dict['conv_block1'])

        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)

        return encoder_pool, encoder


class DecoderBlockRes1B(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        upsample: Tuple,
        momentum: float,
        has_film: bool,
    ):
        r"""Decoder block."""

        super(DecoderBlockRes1B, self).__init__()

        self.kernel_size = kernel_size
        self.stride = upsample

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(in_channels, momentum=momentum)
        # Do not delate the dummy self.bn2. FiLM need self.bn2 to parse the
        # FiLM meta correctly.

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=(0, 0),
            bias=False,
            dilation=(1, 1),
        )

        self.conv_block2 = ConvBlockRes(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            momentum=momentum,
            has_film=has_film,
        )

        self.has_film = has_film

        self.init_weights()

    def init_weights(self):
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(
        self,
        input_tensor: torch.Tensor,
        concat_tensor: torch.Tensor,
        film_dict: Dict,
    ) -> torch.Tensor:
        r"""Forward input feature maps to the decoder block.

        Args:
            input_tensor (torch.Tensor), (B, C_in, T, F)
            film_dict (Dict)

        Returns:
            output (torch.Tensor): (B, C_out, T * upsample, F * upsample)
        """

        b1 = film_dict['beta1']

        x = self.conv1(F.leaky_relu_(self.bn1(input_tensor) + b1))
        # (B, C_out, T * upsample, F * upsample)

        x = torch.cat((x, concat_tensor), dim=1)
        # (B, C_out * 2, T * upsample, F * upsample)

        output = self.conv_block2(x, film_dict['conv_block2'])
        # output: (B, C_out, T * upsample, F * upsample)

        return output


class ResUNet30_Base(nn.Module, Base):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 ) -> None:
        r"""Base separation model.

        Args:
            input_channels (int), audio channels, e.g., 1 | 2
            output_channels (int), audio channels, e.g., 1 | 2
        """

        super(ResUNet30_Base, self).__init__()

        window_size = 2048
        hop_size = 320
        center = True
        pad_mode = "reflect"
        window = "hann"
        momentum = 0.01

        self.output_channels = output_channels

        self.K = 3  # mag, cos, sin

        # This number equals 2^{#encoder_blcoks}
        self.time_downsample_ratio = 2 ** 5

        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)

        self.pre_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.encoder_block1 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block2 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block3 = EncoderBlockRes1B(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block4 = EncoderBlockRes1B(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block5 = EncoderBlockRes1B(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.encoder_block6 = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 2),
            momentum=momentum,
            has_film=True,
        )
        self.conv_block7a = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 1),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block1 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(1, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block2 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block3 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block4 = DecoderBlockRes1B(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block5 = DecoderBlockRes1B(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )
        self.decoder_block6 = DecoderBlockRes1B(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
        )

        self.after_conv = nn.Conv2d(
            in_channels=32,
            out_channels=output_channels * self.K,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        r"""Initialize weights."""
        init_bn(self.bn0)
        init_layer(self.pre_conv)
        init_layer(self.after_conv)

    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (B, input_channels, T, F)
            sp: (B, output_channels, T, F)
            sin_in: (B, output_channels, T, F)
            cos_in: (B, output_channels, T, F)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (B, output_channels, audio_samples)
        """

        x = rearrange(
            input_tensor,
            'b (c k) t f -> b c k t f',
            c=self.output_channels)

        mask_mag = torch.sigmoid(x[:, :, 0, :, :])
        mask_real = torch.tanh(x[:, :, 1, :, :])
        mask_imag = torch.tanh(x[:, :, 2, :, :])

        _, mask_cos, mask_sin = magphase(mask_real, mask_imag)
        # mask_cos, mask_sin: (B, output_channels, T, F)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in * mask_cos - sin_in * mask_sin
        )
        out_sin = (
            sin_in * mask_cos + cos_in * mask_sin
        )
        # out_cos: (B, output_channels, T, F)
        # out_sin: (B, output_channels, T, F)

        # Calculate |Y|.
        out_mag = F.relu_(sp * mask_mag)
        # out_mag: (B, output_channels, T, F)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (B, output_channels, T, F)

        # Reshape to (N, 1, T, F) for ISTFT
        out_real = rearrange(out_real, 'b c t f -> (b c) t f').unsqueeze(1)
        out_imag = rearrange(out_imag, 'b c t f -> (b c) t f').unsqueeze(1)

        # ISTFT
        x = self.istft(out_real, out_imag, audio_length)
        # (B * output_channels, audio_samples)

        # Reshape to (B, output_channels, audio_samples)
        waveform = rearrange(x, '(b c) t -> b c t', c=self.output_channels)

        return waveform

    def forward(self, mixtures, film_dict):
        r"""Forward mixtures and conditions to separate target sources.

        Args:
            input (torch.Tensor): (batch_size, output_channels, segment_samples)

        Outputs:
            output_dict: {
                "waveform": (batch_size, output_channels, segment_samples),
            }
        """

        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(mixtures)
        x = mag

        # Batch normalization
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)   # shape: (B, input_channels, T, F)

        # Pad spectrogram to be evenly divided by downsample ratio
        origin_len = x.shape[2]
        pad_len = (int(np.ceil(x.shape[2] /
                               self.time_downsample_ratio)) *
                   self.time_downsample_ratio -
                   origin_len)
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        # x: (B, input_channels, T, F)

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0: x.shape[-1] - 1]  # (B, input_channels, T, F)

        # UNet
        x = self.pre_conv(x)

        x1_pool, x1 = self.encoder_block1(
            x, film_dict['encoder_block1'])  # x1_pool: (B, 32, T / 2, F / 2)
        x2_pool, x2 = self.encoder_block2(
            x1_pool, film_dict['encoder_block2'])  # x2_pool: (B, 64, T / 4, F / 4)
        x3_pool, x3 = self.encoder_block3(
            x2_pool, film_dict['encoder_block3'])  # x3_pool: (B, 128, T / 8, F / 8)
        # x4_pool: (B, 256, T / 16, F / 16)
        x4_pool, x4 = self.encoder_block4(x3_pool, film_dict['encoder_block4'])
        # x5_pool: (B, 384, T / 32, F / 32)
        x5_pool, x5 = self.encoder_block5(x4_pool, film_dict['encoder_block5'])
        # x6_pool: (B, 384, T / 32, F / 64)
        x6_pool, x6 = self.encoder_block6(x5_pool, film_dict['encoder_block6'])
        x_center, _ = self.conv_block7a(
            x6_pool, film_dict['conv_block7a'])  # (B, 384, T / 32, F / 64)
        # (B, 384, T / 32, F / 32)
        x7 = self.decoder_block1(x_center, x6, film_dict['decoder_block1'])
        # (B, 384, T / 16, F / 16)
        x8 = self.decoder_block2(x7, x5, film_dict['decoder_block2'])
        x9 = self.decoder_block3(
            x8, x4, film_dict['decoder_block3'])  # (B, 256, T / 8, F / 8)
        x10 = self.decoder_block4(
            x9, x3, film_dict['decoder_block4'])  # (B, 128, T / 4, F / 4)
        x11 = self.decoder_block5(
            x10, x2, film_dict['decoder_block5'])  # (B, 64, T / 2, F / 2)
        x12 = self.decoder_block6(
            x11, x1, film_dict['decoder_block6'])  # (B, 32, T, F)

        x = self.after_conv(x12)

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        audio_length = mixtures.shape[2]

        # Convert feature maps to waveform
        separated_audio = self.feature_maps_to_wav(
            input_tensor=x,
            sp=mag,
            sin_in=sin_in,
            cos_in=cos_in,
            audio_length=audio_length,
        )
        # shape:（B, output_channels, segment_samples)

        output_dict = {'waveform': separated_audio}

        return output_dict


class ResUNet30(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 condition_size: int,
                 ) -> None:
        r"""Universal separation model.

        Args:
            input_channels (int), audio channels, e.g., 1 | 2
            output_channels (int), audio channels, e.g., 1 | 2
            condition_size (int), FiLM condition size, e.g., 527 | 2048
        """

        super(ResUNet30, self).__init__()

        self.base = ResUNet30_Base(
            input_channels=input_channels,
            output_channels=output_channels,
        )

        self.film_meta = get_film_meta(
            module=self.base,
        )

        self.film = FiLM(
            film_meta=self.film_meta,
            condition_size=condition_size
        )

    def forward(self, input_dict: Dict) -> Dict:
        r"""Forward mixtures and conditions to separate target sources.

        Args:
            input_dict (Dict): {
                "mixture": (batch_size, audio_channels, audio_samples),
                "condition": (batch_size, condition_dim),
            }

        Returns:
            output_dict (Dict): {
                "waveform": (batch_size, audio_channels, audio_samples)
            }
        """

        mixtures = input_dict['mixture']
        conditions = input_dict['condition']

        film_dict = self.film(
            conditions=conditions,
        )

        output_dict = self.base(
            mixtures=mixtures,
            film_dict=film_dict,
        )

        return output_dict
