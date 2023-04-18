from typing import Any, Callable, Dict

# import pytorch_lightning as pl
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class LitSeparation(pl.LightningModule):
    '''
    def __init__(
        self,
        ss_model: nn.Module,
        anchor_segment_detector,
        anchor_segment_mixer,
        query_condition_extractor,
        loss_function,
        optimizer_type: str,
        learning_rate: float,
        lr_lambda_func,
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
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.lr_lambda_func = lr_lambda_func

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
            'segment': segments[:, None, :],
        }

        self.ss_model.train()
        sep_segment = self.ss_model(input_dict)['waveform']
        # (batch_size, 1, segment_samples)

        output_dict = {
            'segment': sep_segment,
        }

        # Calculate loss.
        loss = self.loss_function(output_dict, target_dict)

        return loss
    '''
    def __init__(
        self,
        ss_model: nn.Module,
        anchor_segment_detector,
        anchor_segment_mixer,
        query_net,
        loss_function,
        optimizer_type: str,
        learning_rate: float,
        lr_lambda_func,
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
        self.query_net = query_net
        self.loss_function = loss_function
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.lr_lambda_func = lr_lambda_func

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

        conditions = self.query_net(
            source=segments,
        )['output']
        
        input_dict = {
            'mixture': mixtures[:, None, :],
            'condition': conditions,
        }

        target_dict = {
            'segment': segments[:, None, :],
        }

        self.ss_model.train()
        sep_segment = self.ss_model(input_dict)['waveform']
        # (batch_size, 1, segment_samples)

        output_dict = {
            'segment': sep_segment,
        }

        # Calculate loss.
        loss = self.loss_function(output_dict, target_dict)

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

        if self.optimizer_type == "AdamW":
            optimizer = optim.AdamW(
                params=self.ss_model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0,
                amsgrad=True,
            )
        else:
            raise NotImplementedError

        scheduler = LambdaLR(optimizer, self.lr_lambda_func)

        output_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

        return output_dict
    
    '''
    def configure_optimizers(self):
        r"""Configure optimizer.
        """
        
        optimizer = optim.AdamW(
            self.ss_model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
        )

        def lr_lambda(step):
            if step < 1000:
                lr_scale = step / 1000
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
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:

        del checkpoint['optimizer_states']

        for key in checkpoint['state_dict'].keys():
            if "anchor_segment_detector" in key:
                del checkpoint['state_dict'][key]

        # from IPython import embed; embed(using=False); os._exit(0)
    '''

def get_model_class(model_type):
    if model_type == 'ResUNet30':
        from casa.models.resunet import ResUNet30
        return ResUNet30

    else:
        raise NotImplementedError
