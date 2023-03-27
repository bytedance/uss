import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorSegmentScorer(nn.Module):
    def __init__(self, 
        segment_frames,
    ):
        super(AnchorSegmentScorer, self).__init__()

        self.segment_frames = segment_frames

        filter_len = self.segment_frames + 1
        self.register_buffer('smooth_filter', torch.ones(1, 1, filter_len) / filter_len)

    def __call__(self, x):
        r"""Calculate smoothed probability, equivalent to the area of 
        probability of anchor segments.

        Args:
            x: (clip_frames,), probability array

        Returns:
            output: (clip_frames,), smoothed probability, equivalent to the 
                area of probability of anchor segments.
        """
        x = F.pad(
            input=x[None, None, :], 
            pad=(self.segment_frames // 2,self.segment_frames // 2),
            mode='replicate'
        )

        output = torch.conv1d(
            input=x, 
            weight=self.smooth_filter, 
            padding=0,
        )
        # (1, 1, clip_frames)

        output = output.flatten()
        # (clip_frames,)

        return output


class AnchorSegmentDetector(nn.Module):
    def __init__(self, 
        sed_model: nn.Module, 
        clip_seconds: float,
        segment_seconds: float,
        frames_per_second: int,
        sample_rate: float,
    ):
        r"""Input a batch of 10-second audio clips, select 2-second anchor 
        segments for creating mixtures. Selected anchor segments will have 
        disjoint audio tagging predictions.

        Args:
            sed_model: nn.Module, pretrained sound event detection system.
            at_model: nn.Module, pretrained audio tagging system.
            segment_seconds, float, e.g., 2.0
            sample_rate: float
            clip_samples: int
        """
        super(AnchorSegmentDetector, self).__init__()

        self.sed_model = sed_model
        self.clip_frames = int(clip_seconds * frames_per_second)
        self.segment_frames = int(segment_seconds * frames_per_second)
        self.hop_samples = sample_rate // frames_per_second
        # self.clip_samples = clip_samples
        # self.augmentor = augmentor
        # self.condition_type = condition_type
        # self.segment_mix_type = segment_mix_type

        # # Smooth filter to smooth sound event detection result.
        self.anchor_segment_scorer = AnchorSegmentScorer(
            segment_frames=self.segment_frames
        )

        # opt_thres = pickle.load(open('opt_thres.pkl', 'rb'))
        # self.register_buffer('opt_thres', torch.Tensor(opt_thres))

    
    def __call__(self, waveforms, class_ids):
        r"""Input a batch of 10-second audio clips, select 2-second anchor 
        segments for creating mixtures. Selected anchor segments will have 
        disjoint audio tagging predictions.

        Args:
            batch_data_dict: dict, e.g., {
                'hdf5_path': (batch_size,),
                'index_in_hdf5': (batch_size,),
                'audio_name': (batch_size,),
                'waveform': (batch_size, clip_samples,), 
                'target': (batch_size, classes_num,),
                'class_id': int,
            }

            full_info: bool, set to True to return full info for creating 
                evaluation meta csv. Set to False to return necessary info for
                training.

        Returns:
            anchor_segment_dict: dict, e.g., {
                'mixture': (new_batch_size, 1, segment_samples),
                'source': (new_batch_size, 1, segment_samples),
                'condition': (new_batch_size, classes_num),
                ...
            }
        """

        batch_size = waveforms.shape[0]

        # Sound event detection
        with torch.no_grad():
            self.sed_model.eval()

            framewise_output = self.sed_model(
                input=waveforms, 
            )['framewise_output']
            # (batch_size, segment_frames, classes_num)

        segments = []   # Will be (new_batch_size, segment_samples)
        bgn_samples = []
        end_samples = []

        # Get candidates of anchor segments using pretrained sound event 
        # detection system.
        for n in range(batch_size):

            # Class ID of the current 10-second clip.
            class_id = class_ids[n]
            # There can be multiple tags in the 10-second clip, but we only 
            # want to find out the when is the 2-second anchor segment that 
            # most likely to contain the class_id-th sound class.

            prob_array = framewise_output[n, :, class_id]
            # (segment_frames,)

            # Smoothed probability, equivalent to the area of probability in 
            # anchor segments.
            anchor_segment_scores = self.anchor_segment_scorer(prob_array)

            if False:
                import soundfile
                soundfile.write(file='_zz{}.wav'.format(n), data=waveforms[n].data.cpu().numpy(), samplerate=32000)
                import matplotlib.pyplot as plt
                from casa.config import IX_TO_LB
                plt.figure()
                plt.plot(anchor_segment_scores.data.cpu().numpy())
                plt.title(IX_TO_LB[class_id])
                plt.ylim(0, 1)
                plt.savefig('_zz{}.pdf'.format(n))

            # Find out the frame with the highest probability. This frames is
            # the centre frame of an anchor segment.
            anchor_index = torch.argmax(anchor_segment_scores)
            
            # Get begin and end samples of an anchor segment.
            bgn_sample, end_sample = self.get_segment_bgn_end_samples(
                anchor_index=anchor_index,
                clip_frames=self.clip_frames,
            )

            segment = waveforms[n, bgn_sample : end_sample]

            segments.append(segment)
            bgn_samples.append(bgn_sample)
            end_samples.append(end_sample)

            # if full_info:
            #     bgn_samples.append(bgn_sample)
            #     end_samples.append(end_sample)

            # Set to True for debugging SED probability and anchor segments.
            if False:
                if n == 5:
                    # from IPython import embed; embed(using=False); os._exit(0)
                    debug_plot_segment(
                        class_id=batch_data_dict['class_id'][n], 
                        waveform=batch_data_dict['waveform'][n], 
                        segment=segments[n], 
                        bgn_sample=bgn_sample, 
                        framewise_output=smoothed_framewise_output)
                    print(IX_TO_LB[class_id])
                    from IPython import embed; embed(using=False); os._exit(0)
                    exit()

        # Mini-batch of 2-second anchor segments.
        segments = torch.stack(segments, dim=0)
        # (batch_size, segment_samples)

        segments_dict = {
            'waveform': segments,
        }

        return segments_dict

    def get_condition(self, at_model, segment, class_id, target, condition_type):

        if condition_type == 'one-hot':
            condition = torch.zeros_like(target)
            condition[class_id] = 1  # (classes_num,)

        elif condition_type == 'multi-hot':
            condition = target  # (classes_num,)

        elif condition_type == 'segment_prediction':
            with torch.no_grad():
                at_model.eval()
                condition = at_model(segment[None, :])['clipwise_output'][0]  # (classes_num,)

        elif condition_type == 'embedding':
            with torch.no_grad():
                at_model.eval()
                condition = at_model(segment[None, :])['embedding'][0]  # (embedding_size,)
            
        else:
            raise NotImplementedError

        return condition

    def get_segment_bgn_end_samples(self, anchor_index, clip_frames):
        r"""Get begin and end samples of an anchor segment.

        Args:
            anchor_index: int, e.g., 196

        Returns:
            bgn_sample: int, e.g., 30720
            end_sample: int, e.g., 94720
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

    def binarize_audio_tagging_predictions(self, audio_tagging_probs):
        r"""Use thresholds to convert audio tag probabilites into binarized 
        outputs.

        Args:
            audio_tagging_probs: (batch_size, classes_num), e.g.,
                [[0.1, 0.8, 0.2, ...],
                 [0.7, 0.6, 0.1, ...],
                 ...]

        Outputs:
            binarized_at_predictions, (batch_size, classes_num), e.g.,
                [[0, 1, 0, ...],
                 [1, 1, 0, ...],
                 ...]
        """
        binarized_at_predictions = (torch.sign(audio_tagging_probs - self.opt_thres[None, :] / 2) + 1) / 2
        return binarized_at_predictions

    def get_indexes_for_mixing(self, binarized_at_predictions):
        r"""Mine anchor segments for creating mixtures. Anchor segments to be 
        mixed should have disjoint predicted tags. Then, return the index of 
        mined anchor segments. 

        Args:
            binarized_at_predictions, (batch_size, classes_num), e.g.,
                [[0, 1, 0, ...],
                 [1, 1, 0, ...],
                 ...]

        Returns:
            new_indexes: list, e.g., [0, 1, 2, 3, ...]
        """
        
        segments_num = binarized_at_predictions.shape[0]

        indexes = list(range(segments_num))
        new_indexes = []

        while indexes:
            i = indexes[0]
            indexes.remove(i)

            for j in indexes:
                segment_i_probs = binarized_at_predictions[i]
                segment_j_probs = binarized_at_predictions[j]

                # Two anchor segments to be mixed should have disjoint 
                # predicted tags.
                if torch.sum(segment_i_probs * segment_j_probs) == 0:
                    new_indexes.append(i)
                    new_indexes.append(j)
                    indexes.remove(j)
                    break
        
        return new_indexes

    def get_energy_ratio(self, segment1, segment2):
        energy1 = energy(segment1)
        energy2 = max(1e-8, energy(segment2))
        ratio = (energy1 / energy2) ** 0.5
        ratio = torch.clamp(ratio, 0.1, 10)

        return ratio