class AnchorSegmentDetector(nn.Module):
    def __init__(self, 
        sed_model: nn.Module, 
        # at_model: nn.Module, 
        segment_seconds: float,
        frames_per_second: int,
        sample_rate: float,
        # clip_samples: int,
        # augmentor,
        # condition_type: str,
        # segment_mix_type: str,
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
        # self.at_model = at_model
        self.segment_frames = int(segment_seconds * frames_per_second)
        self.hop_samples = sample_rate // frames_per_second
        self.clip_samples = clip_samples
        # self.augmentor = augmentor
        # self.condition_type = condition_type
        # self.segment_mix_type = segment_mix_type

        # # Smooth filter to smooth sound event detection result.
        # filter_len = self.segment_frames + 1
        # self.register_buffer('smooth_filter', torch.ones(1, 1, filter_len) / filter_len)

        # opt_thres = pickle.load(open('opt_thres.pkl', 'rb'))
        # self.register_buffer('opt_thres', torch.Tensor(opt_thres))

    
    def __call__(self, waveform):
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

        batch_size = waveform.shape[0]

        # Sound event detection
        with torch.no_grad():
            self.sed_model.eval()

            framewise_output = self.sed_model(
                input=waveform, 
            )['framewise_output']
            # (batch_size, segment_frames, classes_num)

        from IPython import embed; embed(using=False); os._exit(0)

        segments = []   # Will be (new_batch_size, segment_samples)

        if full_info:
            bgn_samples = []
            end_samples = []

        # Get candidates of anchor segments using pretrained sound event 
        # detection system.
        for n in range(batch_size):

            # Class ID of the current 10-second clip.
            class_id = batch_data_dict['class_id'][n]
            # There can be multiple tags in the 10-second clip, but we only 
            # want to find out the when is the 2-second anchor segment that 
            # most likely to contain the class_id-th sound class.

            prob_array = framewise_output[n, :, class_id]
            # (segment_frames,)

            # Smoothed probability, equivalent to the area of probability in 
            # anchor segments.
            smoothed_framewise_output = self.smooth(prob_array)
            # smoothed_framewise_output = prob_array
            # (segment_frames,)

            # Find out the frame with the highest probability. This frames is
            # the centre frame of an anchor segment.
            anchor_index = torch.argmax(smoothed_framewise_output)
            
            # Get begin and end samples of an anchor segment.
            (bgn_sample, end_sample) = self.get_segment_bgn_end_samples(
                anchor_index, self.clip_samples)
            
            segments.append(batch_data_dict['waveform'][n, bgn_sample : end_sample])

            if full_info:
                bgn_samples.append(bgn_sample)
                end_samples.append(end_sample)

            # if class_id == 315:
            #     from IPython import embed; embed(using=False); os._exit(0)
            #     plt.plot(smoothed_framewise_output.data.cpu().numpy())
            #     plt.savefig('_zz.pdf')


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

            # print(bgn_sample, end_sample)
        # from IPython import embed; embed(using=False); os._exit(0)  

        # Mini-batch of 2-second anchor segments.
        segments = torch.stack(segments, dim=0)
        # (batch_size, segment_samples)

        # for i in range(16):
        #     import soundfile
        #     soundfile.write(file='_tmp/zz{:03d}_{}.wav'.format(i, IX_TO_LB[batch_data_dict['class_id'][i]]), data=segments[i].data.cpu().numpy(), samplerate=32000)
        # from IPython import embed; embed(using=False); os._exit(0)

        # Predict audio tagging probabilites of 2-second anchor segments.
        with torch.no_grad():
            self.at_model.eval() 
            segment_at_probs = self.at_model(segments)['clipwise_output']
            # (batch_size, classes_num)

        # Indexes of mined anchor segments for creating mixtures.
        if self.segment_mix_type == "none":
            indexes = np.arange(batch_size)
            
        elif self.segment_mix_type == "avoid_conflict":
            # Convert audio tagging probabilites into binarized outputs.
            binarized_at_predictions = self.binarize_audio_tagging_predictions(segment_at_probs)
            # (batch_size, classes_num)

            indexes = self.get_indexes_for_mixing(binarized_at_predictions)
            # print(indexes)
            # from IPython import embed; embed(using=False); os._exit(0)
        
        else:
            raise NotImplementedError

        if len(indexes) == 0:
            indexes = [0, 1]


        if full_info:
            anchor_segment_dict = {
                'hdf5_path': [],
                'index_in_hdf5': [],
                'audio_name': [],
                'class_id': [],
                'target': [],
                'mixture': [],
                'source': [],
                'condition': [],
                'bgn_sample': [],
                'end_sample': [],
                'mix_rank': []
                }
        else:
            anchor_segment_dict = {
                'mixture': [],
                'source': [],
                'condition': []
                }

        # Build mixtures, sources and conditions.
        for i in range(0, len(indexes), 2):
            
            n1 = indexes[i]
            n2 = indexes[i + 1]

            '''
            from IPython import embed; embed(using=False); os._exit(0)
            import soundfile
            print(IX_TO_LB[batch_data_dict['class_id'][n1]])
            print(IX_TO_LB[batch_data_dict['class_id'][n2]])
            soundfile.write(file='_zz.wav', data=segments[n1].data.cpu().numpy(), samplerate=32000)
            soundfile.write(file='_zz2.wav', data=segments[n2].data.cpu().numpy(), samplerate=32000)

            torch.max(torch.abs(segments[n1]))
            torch.max(torch.abs(segments[n2]))
            '''            

            if True:
                segment1, segment2 = self.augmentor(segments[n1], segments[n2])
                
                condition1 = self.get_condition(
                    at_model=self.at_model, 
                    segment=segment1, 
                    class_id=batch_data_dict['class_id'][n1], 
                    target=batch_data_dict['target'][n1], 
                    condition_type=self.condition_type
                )
                condition2 = self.get_condition(
                    at_model=self.at_model, 
                    segment=segment2, 
                    class_id=batch_data_dict['class_id'][n2], 
                    target=batch_data_dict['target'][n2], 
                    condition_type=self.condition_type
                )
            
            else:
                segment1 = segments[n1]
                segment2 = segments[n2]
                
                condition1 = self.get_condition(
                    at_model=self.at_model, 
                    segment=segment1, 
                    class_id=batch_data_dict['class_id'][n1], 
                    target=batch_data_dict['target'][n1], 
                    condition_type=self.condition_type
                )
                condition2 = self.get_condition(
                    at_model=self.at_model, 
                    segment=segment2, 
                    class_id=batch_data_dict['class_id'][n2], 
                    target=batch_data_dict['target'][n2], 
                    condition_type=self.condition_type
                )

                segment1, segment2 = self.augmentor(segment1, segment2)

            mixture = segment1 + segment2

            '''
            if i == 4:
                import soundfile
                print(IX_TO_LB[np.argmax(condition1.data.cpu().numpy())])
                print(IX_TO_LB[np.argmax(condition2.data.cpu().numpy())])
                soundfile.write(file='_zz.wav', data=segment1.data.cpu().numpy(), samplerate=32000)
                soundfile.write(file='_zz2.wav', data=segment2.data.cpu().numpy(), samplerate=32000)
                soundfile.write(file='_zz3.wav', data=(segment1+segment2).data.cpu().numpy(), samplerate=32000)
                # from audioset_source_separation.utils import energy
                from IPython import embed; embed(using=False); os._exit(0)
            '''

            # mixutres
            anchor_segment_dict['mixture'].append(mixture)
            anchor_segment_dict['mixture'].append(mixture)
            
            # sources
            anchor_segment_dict['source'].append(segment1)
            anchor_segment_dict['source'].append(segment2)
            
            # soft conditions
            anchor_segment_dict['condition'].append(condition1)
            anchor_segment_dict['condition'].append(condition2)
            
            if full_info:
                # hdf5 file paths
                anchor_segment_dict['hdf5_path'].append(batch_data_dict['hdf5_path'][n1])
                anchor_segment_dict['hdf5_path'].append(batch_data_dict['hdf5_path'][n2])

                # indexes in hdf5 files
                anchor_segment_dict['index_in_hdf5'].append(batch_data_dict['index_in_hdf5'][n1])
                anchor_segment_dict['index_in_hdf5'].append(batch_data_dict['index_in_hdf5'][n2])

                # audio names
                anchor_segment_dict['audio_name'].append(batch_data_dict['audio_name'][n1])
                anchor_segment_dict['audio_name'].append(batch_data_dict['audio_name'][n2])
                
                # class ids
                anchor_segment_dict['class_id'].append(batch_data_dict['class_id'][n1])
                anchor_segment_dict['class_id'].append(batch_data_dict['class_id'][n2])

                # audio tag labels
                anchor_segment_dict['target'].append(batch_data_dict['target'][n1])
                anchor_segment_dict['target'].append(batch_data_dict['target'][n2])

                # bgn samples
                anchor_segment_dict['bgn_sample'].append(bgn_samples[n1])
                anchor_segment_dict['bgn_sample'].append(bgn_samples[n2])

                # end samples
                anchor_segment_dict['end_sample'].append(end_samples[n1])
                anchor_segment_dict['end_sample'].append(end_samples[n2])

                # ranks
                anchor_segment_dict['mix_rank'].append(0)
                anchor_segment_dict['mix_rank'].append(1)

        # for i in range(16):
        #     import soundfile
        #     soundfile.write(file='_tmp/zz{:03d}.wav'.format(i), data=anchor_segment_dict['mixture'][i].data.cpu().numpy(), samplerate=32000)

        # Tackle the situation if no anchor segments are mined. This happens
        # very rarely.
        # if len(anchor_segment_dict['mixture']) == 0:
        if len(indexes) == 0:

            for n1 in range(2):
                anchor_segment_dict['mixture'].append(segments[n1])
                anchor_segment_dict['source'].append(segments[n1])
                condition = self.get_condition(self.at_model, segments[n1], self.condition_type)
                anchor_segment_dict['condition'].append(condition)

                if full_info:
                    anchor_segment_dict['hdf5_path'].append(batch_data_dict['hdf5_path'][n1])
                    anchor_segment_dict['index_in_hdf5'].append(batch_data_dict['index_in_hdf5'][n1])
                    anchor_segment_dict['audio_name'].append(batch_data_dict['audio_name'][n1])
                    anchor_segment_dict['class_id'].append(batch_data_dict['class_id'][n1])
                    anchor_segment_dict['target'].append(batch_data_dict['target'][n1])
                    anchor_segment_dict['bgn_sample'].append(bgn_samples[n1])
                    anchor_segment_dict['end_sample'].append(end_samples[n1])
                    anchor_segment_dict['mix_rank'].append(0)
                    anchor_segment_dict['mix_rank'].append(1)

        if full_info:
            keys = ['mixture', 'source', 'condition', 'target', 'bgn_sample', 'end_sample']
        else:
            keys = ['mixture', 'source', 'condition']

        for key in keys:
            anchor_segment_dict[key] = torch.stack(anchor_segment_dict[key], dim=0)
            
        for key in ['mixture', 'source']:
            anchor_segment_dict[key] = anchor_segment_dict[key][:, None, :]
        # (batch_size, channels_num, segment_samples)
        
        # from IPython import embed; embed(using=False); os._exit(0)
        # soundfile.write(file='_zz.wav', data=anchor_segment_dict['mixture'][3, 0].data.cpu().numpy(), samplerate=32000)

        return anchor_segment_dict

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

    def smooth(self, x):
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
        
        output = output[0, 0, :]
        # (clip_frames,)

        return output

    def get_segment_bgn_end_samples(self, anchor_index, clip_samples):
        r"""Get begin and end samples of an anchor segment.

        Args:
            anchor_index: int, e.g., 196

        Returns:
            bgn_sample: int, e.g., 30720
            end_sample: int, e.g., 94720
        """

        bgn_frame = anchor_index - self.segment_frames // 2
        # E.g., 96

        bgn_sample = bgn_frame * self.hop_samples
        # E.g., 30720

        segment_samples = self.segment_frames * self.hop_samples
        # E.g., 64000
        
        bgn_sample = torch.clamp(bgn_sample, 0, clip_samples - segment_samples)

        end_sample = bgn_sample + segment_samples
        # E.g., 94720

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
