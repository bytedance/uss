from typing import Any, Callable, Dict

# import pytorch_lightning as pl
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class LitSeparation(pl.LightningModule):
    def __init__(
        self,
        ss_model: nn.Module,
        anchor_segment_detector,
        anchor_segment_mixer,
        query_condition_extractor,
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
        self.ss_model = ss_model
        self.anchor_segment_detector = anchor_segment_detector
        self.anchor_segment_mixer = anchor_segment_mixer
        self.query_condition_extractor = query_condition_extractor
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.lr_lambda = lr_lambda

    def forward(self, x):
        pass

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
            class_ids=batch_data_dict['class_id'],
        )
        
        mixtures, segments = self.anchor_segment_mixer(
            waveforms=segments_dict['waveform'],
        )

        conditions = self.query_condition_extractor(
            segments=segments,
        )

        input_dict = {
            'mixture': mixtures[:, None, :],
            'condition': conditions,
        }

        target_dict = {
            'segment': segments,
        }

        self.ss_model.train()
        sep_segment = self.ss_model(input_dict)['waveform']
        sep_segment = sep_segment.squeeze()
        # (batch_size, 1, segment_samples)

        output_dict = {
            'segment': sep_segment,
        }

        # Calculate loss.
        loss = self.loss_function(output_dict, target_dict)

        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({"train_loss": loss})

        return loss

    def test_step(self, batch, batch_idx):
        pass


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

    # def validation_step(self, batch, batch_idx):
    #     from IPython import embed; embed(using=False); os._exit(0)
    
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
