import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorSegmentDetector(nn.Module):
    def __init__(self,
                 sed_model: nn.Module,
                 clip_seconds: float,
                 segment_seconds: float,
                 frames_per_second: int,
                 sample_rate: float,
                 detect_mode: str,
                 ) -> None:
        r"""The anchor segment detector is used to detect 2-second anchor
        segments from 10-second weakly labelled audio clips during training.

        Args:
            sed_model (nn.Module): pretrained sound event detection model
            clip_seconds (float): audio clip duration, e.g., 10.
            segment_seconds (float): anchor segment duration, e.g., 2.
            frames_per_second (int):, e.g., 100
            sample_rate (int)
            detect_mode (str), "max_area" | "random"

        Returns:
            None
        """
        super(AnchorSegmentDetector, self).__init__()

        assert detect_mode in ["max_area", "random"]

        self.sed_model = sed_model
        self.clip_frames = int(clip_seconds * frames_per_second)
        self.segment_frames = int(segment_seconds * frames_per_second + 1)
        self.hop_samples = sample_rate // frames_per_second
        self.sample_rate = sample_rate
        self.detect_mode = detect_mode

        # Used to the area under the probability curve of anchor segments
        self.anchor_segment_scorer = AnchorSegmentScorer(
            segment_frames=self.segment_frames,
        )

    def __call__(
        self,
        waveforms: torch.Tensor,
        class_ids: List,
        debug: bool = False,
    ) -> Dict:
        r"""Input a batch of 10-second audio clips. Mine 2-second anchor
        segments.

        Args:
            waveforms (torch.Tensor): (batch_size, clip_samples)
            class_ids (List): (batch_size,)
            debug (bool)

        Returns:
            segments_dict (Dict): e.g., {
                "waveform": (batch_size, segment_samples),
                "class_id": (batch_size,),
                "bgn_sample": (batch_size,),
                "end_sample": (batch_size,),
            }
        """

        batch_size, _ = waveforms.shape

        # Sound event detection
        with torch.no_grad():
            self.sed_model.eval()

            framewise_output = self.sed_model(
                input=waveforms,
            )['framewise_output']
            # (batch_size, clip_frames, classes_num)

        segments = []
        bgn_samples = []
        end_samples = []

        # Detect the anchor segments of a mini-batch of audio clips one by one
        for n in range(batch_size):

            # Class ID of the current 10-second clip.
            class_id = class_ids[n]
            # There can be multiple tags in the 10-second clip. We only detect
            # the anchor segment of #class_id

            prob_array = framewise_output[n, :, class_id]
            # shape: (segment_frames,)

            if self.detect_mode == "max_area":

                # The area of the probability curve in anchor segments
                anchor_segment_scores = self.anchor_segment_scorer(
                    prob_array=prob_array,
                )

                anchor_index = torch.argmax(anchor_segment_scores)

                if debug:
                    _debug_plot_anchor_segment(
                        waveform=waveforms[0],
                        anchor_segment_scores=anchor_segment_scores,
                        class_id=class_id,
                    )

            elif self.detect_mode == "random":
                anchor_index = random.randint(0, self.segment_frames - 1)
                anchor_index = torch.tensor(anchor_index).to(waveforms.device)

            else:
                raise NotImplementedError

            # Get begin and end samples of an anchor segment.
            bgn_sample, end_sample = self.get_segment_bgn_end_samples(
                anchor_index=anchor_index,
                clip_frames=self.clip_frames,
            )

            segment = waveforms[n, bgn_sample: end_sample]

            segments.append(segment)
            bgn_samples.append(bgn_sample)
            end_samples.append(end_sample)

        segments = torch.stack(segments, dim=0)
        # (batch_size, segment_samples)

        bgn_samples = torch.stack(bgn_samples, dim=0)
        end_samples = torch.stack(end_samples, dim=0)

        segments_dict = {
            'waveform': segments,
            'class_id': class_ids,
            'bgn_sample': bgn_samples,
            'end_sample': end_samples,
        }

        return segments_dict

    def get_segment_bgn_end_samples(
        self,
        anchor_index: int,
        clip_frames: int
    ) -> Tuple[int, int]:
        r"""Get begin and end samples of an anchor segment.

        Args:
            anchor_index (torch.int): e.g., 155

        Returns:
            bgn_sample (torch.int), e.g., 17600
            end_sample (torch.int): e.g., 81600
        """

        anchor_index = torch.clamp(
            input=anchor_index,
            min=self.segment_frames // 2,
            max=clip_frames - self.segment_frames // 2,
        )

        bgn_frame = anchor_index - self.segment_frames // 2
        end_frame = anchor_index + self.segment_frames // 2

        bgn_sample = bgn_frame * self.hop_samples
        end_sample = end_frame * self.hop_samples

        return bgn_sample, end_sample


def _debug_plot_anchor_segment(
    waveform: torch.Tensor,
    anchor_segment_scores: torch.Tensor,
    class_id: int,
) -> None:
    r"""For debug only. Plot anchor segment prediction."""

    import os

    import matplotlib.pyplot as plt
    import soundfile

    from uss.config import IX_TO_LB
    sample_rate = 32000

    n = 0
    audio_path = os.path.join("_debug_anchor_segment{}.wav".format(n))
    fig_path = os.path.join("_debug_anchor_segment{}.pdf".format(n))

    soundfile.write(
        file=audio_path,
        data=waveform.data.cpu().numpy(),
        samplerate=sample_rate,
    )
    print("Write out to {}".format(audio_path))

    plt.figure()
    plt.plot(anchor_segment_scores.data.cpu().numpy())
    plt.title(IX_TO_LB[class_id])
    plt.ylim(0, 1)
    plt.savefig(fig_path)
    print("Write out to {}".format(fig_path))

    os._exit(0)


class AnchorSegmentScorer(nn.Module):
    def __init__(self,
                 segment_frames: int,
                 ) -> None:
        r"""Calculate the area under the probabiltiy curve of an anchor segment.

        Args:
            segment_frames (int)

        Returns:
            None
        """

        super(AnchorSegmentScorer, self).__init__()

        self.segment_frames = segment_frames

        filter_len = self.segment_frames
        self.register_buffer(
            'smooth_filter', torch.ones(
                1, 1, filter_len) / filter_len)

    def __call__(self, prob_array: torch.Tensor):
        r"""Calculate the area under the probabiltiy curve of an anchor segment.

        Args:
            prob_array (torch.Tensor): (clip_frames,), sound event
                detection probability of a specific sound class.

        Returns:
            output: (clip_frames,), smoothed probability, equivalent to the
                area of probability of anchor segments.
        """

        x = F.pad(
            input=prob_array[None, None, :],
            pad=(self.segment_frames // 2, self.segment_frames // 2),
            mode='replicate'
        )
        # shape: (1, 1, clip_frames)

        output = torch.conv1d(
            input=x,
            weight=self.smooth_filter,
            padding=0,
        )
        # shape: (1, 1, clip_frames)

        output = output.squeeze(dim=(0, 1))
        # (clip_frames,)

        return output
