import os
from typing import Dict, List, Optional, NoReturn

import numpy as np
import h5py
import torch
import pytorch_lightning as L
from torch.utils.data import DataLoader

# from audioset_source_separation.data.samplers import BalancedSampler, DistributedSamplerWrapper
from casa.utils import int16_to_float32
from casa.data.samplers import DistributedSamplerWrapper#, BatchSampler, BatchSampler2
# from casa.data.datasets import Dataset


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_sampler: object,
        train_dataset: object,
        num_workers: int,
    ):
        r"""Data module. To get one batch of data:

        code-block:: python

            data_module.setup()

            for batch_data_dict in data_module.train_dataloader():
                print(batch_data_dict.keys())
                break

        Args:
            train_sampler: Sampler object
            train_dataset: Dataset object
            num_workers: int
            distributed: bool
        """
        super().__init__()
        self._train_sampler = train_sampler
        self._train_dataset = train_dataset
        self.num_workers = num_workers
        # self.distributed = distributed
        self.collate_fn = collate_fn

        # self._train_sampler = BatchSampler(BalancedSampler(), batch_size=16, drop_last=True)

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage: Optional[str] = None) -> NoReturn:
        r"""called on every device."""

        # make assignments here (val/train/test split)
        # called on every process in DDP

        # SegmentSampler is used for selecting segments for training.
        # On multiple devices, each SegmentSampler samples a part of mini-batch
        # data.
        self.train_dataset = self._train_dataset
        
        self.train_sampler = DistributedSamplerWrapper(self._train_sampler)
        # self.train_sampler = self._train_sampler
        
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        r"""Get train loader."""
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

        return train_loader

    def val_dataloader(self):
        # val_split = Dataset(...)
        # return DataLoader(val_split)
        pass

    def test_dataloader(self):
        # test_split = Dataset(...)
        # return DataLoader(test_split)
        pass

    def teardown(self):
        # clean up after fit or test
        # called on every process in DDP
        pass

    


class Dataset:
    def __init__(self):
        r"""This class takes the meta of an audio clip as input, and return 
        the waveform, target and other information of the audio clip. This 
        class is used by DataLoader.
        """
        pass
    
    def __getitem__(self, meta):
        """Load waveform, target and other information of an audio clip.
        
        Args:
            meta: dict, {
                'hdf5_path': str, 
                'index_in_hdf5': int,
                'class_id': int
            }

        Returns: 
            data_dict: {
                'hdf5_path': str,
                'index_in_hdf5': int,
                'audio_name': str, 
                'waveform': (clip_samples,), 
                'target': (classes_num,),
                'class_id': int,
            }
        """
        
        hdf5_path = meta['hdf5_path']
        index_in_hdf5 = meta['index_in_hdf5']
        class_id = meta['class_id']

        with h5py.File(hdf5_path, 'r') as hf:

            audio_name = hf['audio_name'][index_in_hdf5].decode()

            waveform = int16_to_float32(hf['waveform'][index_in_hdf5])
            waveform = waveform
            # (clip_samples,)

            target = hf['target'][index_in_hdf5].astype(np.float32)
            # (classes_num,)

        data_dict = {
            'hdf5_path': hdf5_path,
            'index_in_hdf5': index_in_hdf5,
            'audio_name': audio_name, 
            'waveform': waveform, 
            'target': target, 
            'class_id': class_id, 
        }

        return data_dict

    def __len__(self):
        return 10000



def collate_fn(list_data_dict):
    r"""Collate mini-batch data to inputs and targets for training.

    Args:
        list_data_dict: e.g., [
            {'vocals': (channels_num, segment_samples), 
             'accompaniment': (channels_num, segment_samples),
             'mixture': (channels_num, segment_samples)
            },
            {'vocals': (channels_num, segment_samples),
             'accompaniment': (channels_num, segment_samples),
             'mixture': (channels_num, segment_samples)
            },
            ...]

    Returns:
        data_dict: e.g. {
            'vocals': (batch_size, channels_num, segment_samples),
            'accompaniment': (batch_size, channels_num, segment_samples),
            'mixture': (batch_size, channels_num, segment_samples)
            }
    """
    
    data_dict = {}
    for key in list_data_dict[0].keys():
    # for key in ['waveform']:
        # try:
        data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
        # except:
        #     from IPython import embed; embed(using=False); os._exit(0)

        if str(data_dict[key].dtype) in ['float32']:
            data_dict[key] = torch.Tensor(data_dict[key])

    return data_dict
