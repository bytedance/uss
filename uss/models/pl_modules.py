from typing import Callable, Dict

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class LitSeparation(pl.LightningModule):
    def __init__(
        self,
        ss_model: nn.Module,
        anchor_segment_detector: nn.Module,
        anchor_segment_mixer: nn.Module,
        query_net: nn.Module,
        loss_function: Callable,
        optimizer_type: str,
        learning_rate: float,
        lr_lambda_func: Callable,
    ) -> None:
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            ss_model (nn.Module): universal source separation module
            anchor_segment_detector (nn.Module): used to detect anchor segments
                from audio clips
            anchor_segment_mixer (nn.Module): used to mix segments into mixtures
            query_net (nn.Module): used to extract conditions for separation
            loss_function (Callable): loss function to train the separation model
            optimizer_type (str): e.g., "AdamW"
            learning_rate (float)
            lr_lambda_func (Callable), learning rate scaler
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

    def training_step(
        self,
        batch_data_dict: Dict,
        batch_idx: int
    ) -> torch.float:
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed on multiple
        devices (if there are) for parallel training.

        Args:
            batch_data_dict (Dict): e.g. {
                'hdf5_path': (batch_size,),
                'index_in_hdf5': (batch_size,),
                'audio_name': (batch_size,),
                'waveform': (batch_size,),
                'target': (batch_size,),
                'class_id': (batch_size,),
            }
            batch_idx: int

        Returns:
            loss (torch.float): loss function of this mini-batch
        """

        # Mine anchor segments from audio clips
        segments_dict = self.anchor_segment_detector(
            waveforms=batch_data_dict['waveform'],
            class_ids=batch_data_dict['class_id'],
        )
        # segments_dict: {
        #     "waveform": (batch_size, segment_samples),
        #     "class_id": (batch_size,),
        #     "bgn_sample": (batch_size,),
        #     "end_sample": (batch_size,),
        # }

        # Mix segments into mixtures and execute energy augmentation
        mixtures, segments = self.anchor_segment_mixer(
            waveforms=segments_dict['waveform'],
        )
        # mixtures: (batch_size, segment_samples)
        # segments: (batch_size, segment_samples)

        # Use query net to calculate conditional embedding
        conditions = self.query_net(
            source=segments,
        )['output']
        # conditions: (batch_size, condition_dim)

        input_dict = {
            'mixture': mixtures[:, None, :],
            'condition': conditions,
        }

        target_dict = {
            'segment': segments[:, None, :],
        }

        # Do separation using mixtures and conditions as input
        self.ss_model.train()
        sep_segment = self.ss_model(input_dict)['waveform']
        # sep_segment: (batch_size, 1, segment_samples)

        output_dict = {
            'segment': sep_segment,
        }

        # Calculate loss
        loss = self.loss_function(output_dict, target_dict)

        return loss

    def configure_optimizers(self) -> Dict:
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


def get_model_class(model_type: str) -> nn.Module:
    r"""Get separation module by model_type."""

    if model_type == 'ResUNet30':
        from uss.models.resunet import ResUNet30
        return ResUNet30

    else:
        raise NotImplementedError
