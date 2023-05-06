from typing import Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

from uss.data.samplers import DistributedSamplerWrapper


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_sampler: object,
        train_dataset: object,
        num_workers: int,
    ) -> None:
        r"""PyTorch Lightning Data module. A wrapper of the DataLoader. Can be
        used to yield mini-batches of train, validation, and test data.

        Args:
            train_sampler (Sampler object)
            train_dataset (Dataset object)
            num_workers: int

        Returns:
            None

        Examples::
            >>> data_module.setup()
            >>> for batch_data_dict in datamodule.train_dataloader():
            >>>     print(batch_data_dict.keys())
            >>>     break
        """

        super().__init__()
        self._train_sampler = train_sampler
        self._train_dataset = train_dataset
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def setup(self, stage: Optional[str] = None) -> None:
        r"""called on every GPU."""

        self.train_dataset = self._train_dataset

        # The sampler yields a part of mini-batch meta on each device
        self.train_sampler = DistributedSamplerWrapper(self._train_sampler)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        r"""Get train loader."""

        if self.num_workers > 0:
            persistent_workers = True
        else:
            persistent_workers = False

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )

        return train_loader

    def val_dataloader(self):
        r"""We use `uss.callbacks.evaluate` to evaluate on the train / test
        dataset"""
        pass


def collate_fn(list_data_dict: List[Dict]) -> Dict:
    r"""Collate a mini-batch of data.

    Args:
        list_data_dict (List[Dict]): e.g., [
            {"hdf5_path": "xx/balanced_train.h5",
             "index_in_hdf5": 11072,
             ...},
        ...]

    Returns:
        data_dict (Dict): e.g., {
            "hdf5_path": ["xx/balanced_train.h5", "xx/balanced_train.h5", ...]
            "index_in_hdf5": [11072, 17251, ...],
        }
    """

    data_dict = {}

    for key in list_data_dict[0].keys():

        data_dict[key] = np.array([data_dict[key]
                                  for data_dict in list_data_dict])

        if str(data_dict[key].dtype) in ["float32"]:
            data_dict[key] = torch.Tensor(data_dict[key])

    return data_dict
