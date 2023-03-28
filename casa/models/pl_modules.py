from typing import Any, Callable, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class LitModel(pl.LightningModule):
    def __init__(
        self,
        sed_model: nn.Module,
        at_model: nn.Module,
        ss_model: nn.Module,
        anchor_segment_detector,
        anchor_segment_mixer,
        query_condition_extractor,
        loss_function,
        learning_rate: float,
        # lr_lambda,
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            ss_model: nn.Module
            anchor_segment_detector: nn.Module
            loss_function: function or object
            learning_rate: float
            lr_lambda: function
        """

        super().__init__()
        # self.batch_data_preprocessor = batch_data_preprocessor
        self.sed_model = sed_model
        self.at_model = at_model
        self.ss_model = ss_model
        self.anchor_segment_detector = anchor_segment_detector
        self.anchor_segment_mixer = anchor_segment_mixer
        self.query_condition_extractor = query_condition_extractor
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        # self.lr_lambda = lr_lambda

    def training_step(self, batch_data_dict, batch_idx):
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch_data_dict: e.g. {
                'hdf5_path': (batch_size,),
                'index_in_hdf5': (batch_size),
                'audio_name': (batch_size),
                'waveform': (batch_size),
                'target': (batch_size),
                'class_id': (batch_size),
            }
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """

        segments_dict = self.anchor_segment_detector(
            waveforms=batch_data_dict['waveform'],
            class_ids=batch_data_dict['class_id'])
        
        mixtures, segments = self.anchor_segment_mixer(
            waveforms=segments_dict['waveform'],
        )

        conditions = self.query_condition_extractor(
            segments=segments,
        )

        input_dict = {
            'mixture': mixtures,
            'condition': conditions,
        }

        target_dict = {
            'segment': segments
        }

        self.ss_model.train()
        outputs = self.ss_model(input_dict)['waveform']
        # (batch_size, 1, segment_samples)

        from IPython import embed; embed(using=False); os._exit(0)

        # Calculate loss.
        loss = self.loss_function(outputs, sources)

        # Compensate for mini-batch sizes are different caused by anchor 
        # segment mining.
        loss = loss / len(mixtures) * batch_size

        '''
        from IPython import embed; embed(using=False); os._exit(0)
        import soundfile
        soundfile.write(file='_zz.wav', data=mixtures.data.cpu().numpy()[3, 0], samplerate=32000)
        soundfile.write(file='_zz2.wav', data=sources.data.cpu().numpy()[6, 0], samplerate=32000)
        '''

        return loss
    
    '''
    def configure_optimizers(self):
        r"""Configure optimizer.
        """

        optimizer = optim.Adam(
            self.ss_model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
        )

        scheduler = {
            'scheduler': LambdaLR(optimizer, self.lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [scheduler]
    '''
    
    def configure_optimizers(self):
        r"""Configure optimizer.
        """

        optimizer = optim.Adam(
            self.ss_model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
        )

        def lr_lambda(step):
            if 0 <= step < 10000:
                lr_scale = 0.001
            elif 10000 <= step < 20000:
                lr_scale = 0.01
            elif 20000 <= step < 30000:
                lr_scale = 0.1
            else:
                lr_scale = 1

            return lr_scale

        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [scheduler]
    
    '''
    def configure_optimizers(self):
        r"""Configure optimizer.
        """

        optimizer = optim.Adam(
            self.ss_model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
        )

        def lr_lambda(step):
            if 0 <= step < 1000:
                lr_scale = 0.001
            elif 1000 <= step < 2000:
                lr_scale = 0.01
            elif 2000 <= step < 3000:
                lr_scale = 0.1
            else:
                lr_scale = 1

            return lr_scale

        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [scheduler]
    '''
    '''
    def configure_optimizers(self):
        r"""Configure optimizer.
        """
        
        optimizer = optim.Adam(
            self.ss_model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
        )

        def lr_lambda(step):
            if step < 30000:
                lr_scale = ((step // 100) * 100) / 30000
            else:
                lr_scale = 1
            return lr_scale

        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [scheduler]
    '''


class LitSourceSeparationTrainableAt(pl.LightningModule):
    def __init__(
        self,
        # at_model: nn.Module,
        ss_model: nn.Module,
        batch_data_preprocessor,
        emb_model,
        loss_function,
        learning_rate: float,
        lr_lambda,
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            ss_model: nn.Module
            anchor_segment_detector: nn.Module
            loss_function: function or object
            learning_rate: float
            lr_lambda: function
        """

        super().__init__()
        self.batch_data_preprocessor = batch_data_preprocessor
        # self.at_model = at_model
        self.ss_model = ss_model
        self.emb_model = emb_model
        # self.anchor_segment_detector = anchor_segment_detector
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.lr_lambda = lr_lambda

    def training_step(self, batch_data_dict, batch_idx):
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch_data_dict: e.g. {
                'hdf5_path': (batch_size,),
                'index_in_hdf5': (batch_size),
                'audio_name': (batch_size),
                'waveform': (batch_size),
                'target': (batch_size),
                'class_id': (batch_size),
            }
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """
        
        # for i in range(16):
        #     import soundfile
        #     soundfile.write(file='_tmp/zz{:03d}.wav'.format(i), data=batch_data_dict['waveform'][i].data.cpu().numpy(), samplerate=32000)
        
        key = list(batch_data_dict.keys())[0]
        batch_size = batch_data_dict[key].shape[0]
        # batch_size = batch_data_dict['waveform'].shape[0]
        
        # segment_data_dict = self.anchor_segment_detector.detect_anchor_segments(
        #     batch_data_dict, full_info=False)
        # E.g., {
        #     'mixture': (batch_size, 1, segment_samples),
        #     'source': (batch_size, 1, segment_samples),
        #     'condition': (batch_size, classes_num),
        # }

        input_dict, target_dict = self.batch_data_preprocessor(batch_data_dict)

        # mixtures = segment_data_dict['mixture']
        mixtures = input_dict['waveform']
        # mixtures = input_dict['waveform']
        # (batch_size, 1, segment_samples)

        # conditions = input_dict['condition']
        # (batch_size, classes_num)

        sources = target_dict['source']
        # (batch_size, 1, segment_samples)

        emb_output_dict = self.emb_model(sources[:, 0, :])
        conditions = emb_output_dict['embedding']
        at_clipwise_output = emb_output_dict['clipwise_output']
        input_dict['condition'] = conditions

        self.ss_model.train()
        # outputs = self.ss_model(mixtures, conditions)['waveform']
        outputs = self.ss_model(input_dict)['waveform']
        # (batch_size, 1, segment_samples)

        # print(torch.mean(torch.abs(self.emb_model.conv_block6.conv1.weight)).item())
        
        # Calculate loss.
        if self.loss_function.__name__ in ['l1_wav_bce_at']:
            output_dict = {
                'wav_output': outputs,
                'at_clipwise_output': at_clipwise_output,
            }
            target_dict = {
                'wav_target': sources,
                'at_target': batch_data_dict['target'],
            }
            loss = self.loss_function(output_dict, target_dict)
        else:
            loss = self.loss_function(outputs, sources)

        # Compensate for mini-batch sizes are different caused by anchor 
        # segment mining.
        loss = loss / len(mixtures) * batch_size

        return loss
    
    def configure_optimizers(self):
        r"""Configure optimizer.
        """

        optimizer = optim.Adam(
            list(self.ss_model.parameters()) + list(self.emb_model.parameters()),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
        )

        def lr_lambda(step):
            if 0 <= step < 10000:
                lr_scale = 0.001
            elif 10000 <= step < 20000:
                lr_scale = 0.01
            elif 20000 <= step < 30000:
                lr_scale = 0.1
            else:
                lr_scale = 1

            return lr_scale

        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [scheduler]
    '''
    def configure_optimizers(self):
        r"""Configure optimizer.
        """

        optimizer = optim.Adam(
            list(self.ss_model.parameters()) + list(self.emb_model.parameters()),
            # self.ss_model.parameters(),
            # self.emb_model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
        )

        def lr_lambda(step):
            return 1.

        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [scheduler]
    '''

def get_model_class(model_type):
    if model_type == 'CondUNet':
        from audioset_source_separation.models.cond_unet import CondUNet
        return CondUNet

    elif model_type == "CondUNetSubbandTime":
        from audioset_source_separation.models.cond_unet_subbandtime import CondUNetSubbandTime
        return CondUNetSubbandTime

    elif model_type == "CondResUNetSubbandTime":
        from audioset_source_separation.models.cond_resunet_subbandtime import CondResUNetSubbandTime
        return CondResUNetSubbandTime

    else:
        raise NotImplementedError('Incorrect model_type!')
