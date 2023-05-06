import logging
import time
from typing import Dict, List

import h5py
import numpy as np
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class Base:
    def __init__(self,
                 indexes_hdf5_path: str,
                 batch_size: int,
                 steps_per_epoch: int,
                 random_seed: int,
                 ):
        r"""Base class of train samplers.

        Args:
            indexes_hdf5_path (str)
            batch_size (int)
            steps_per_epoch (int)
            random_seed (int)

        Returns:
            None
        """

        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)
        self.steps_per_epoch = steps_per_epoch

        # Load targets of training data
        load_time = time.time()

        with h5py.File(indexes_hdf5_path, 'r') as hf:
            self.audio_names = [audio_name.decode()
                                for audio_name in hf['audio_name'][:]]
            self.hdf5_paths = [hdf5_path.decode()
                               for hdf5_path in hf['hdf5_path'][:]]
            self.indexes_in_hdf5 = hf['index_in_hdf5'][:]
            self.targets = hf['target'][:].astype(np.float32)
            # self.targets: (audios_num, classes_num)

        self.audios_num, self.classes_num = self.targets.shape

        logging.info('Training number: {}'.format(self.audios_num))
        logging.info(
            'Load target time: {:.3f} s'.format(
                time.time() - load_time))

        # Number of training samples of different sound classes
        self.samples_num_per_class = np.sum(self.targets, axis=0)

        logging.info('samples_num_per_class: {}'.format(
            self.samples_num_per_class.astype(np.int32)))


class BalancedSampler(Base, Sampler):
    def __init__(self,
                 indexes_hdf5_path: str,
                 batch_size: int,
                 steps_per_epoch: int,
                 random_seed: int = 1234,
                 ) -> None:
        r"""Balanced sampler. Generate mini-batches meta for training. Data are
        evenly sampled from different sound classes.

        Args:
            indexes_hdf5_path (str)
            batch_size (int)
            steps_per_epoch (int)
            random_seed (int)

        Returns:
            None
        """

        super(BalancedSampler, self).__init__(
            indexes_hdf5_path=indexes_hdf5_path,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            random_seed=random_seed,
        )

        # Training indexes of all sound classes. E.g.:
        # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
        self.indexes_per_class = []

        for k in range(self.classes_num):
            self.indexes_per_class.append(
                np.where(self.targets[:, k] == 1)[0])

        # Shuffle indexes
        for k in range(self.classes_num):
            self.random_state.shuffle(self.indexes_per_class[k])

        self.queue = []  # Contains sound class IDs

        self.pointers_of_classes = [0] * self.classes_num

    def expand_queue(self, queue) -> List:
        r"""Append more class IDs to the queue.

        Args:
            queue: List, e.g., [431, 73]

        Returns:
            queue: List, e.g., [431, 73, 2, 54, 379, ...]
        """

        classes_set = np.arange(self.classes_num).tolist()
        self.random_state.shuffle(classes_set)
        queue.extend(classes_set)
        return queue

    def __iter__(self) -> List[Dict]:
        r"""Yield mini-batch meta.

        Args:
            None

        Returns:
            batch_meta: e.g.: [
                {"audio_name": "YfWBzCRl6LUs.wav",
                 "hdf5_path": "xx/balanced_train.h5",
                 "index_in_hdf5": 15734,
                 "target": [0, 1, 0, 0, ...]},
            ...]
        """

        batch_size = self.batch_size

        while True:

            batch_meta = []

            while len(batch_meta) < batch_size:

                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)

                # Pop a class ID and get the audio index
                class_id = self.queue.pop(0)
                pointer = self.pointers_of_classes[class_id]
                self.pointers_of_classes[class_id] += 1
                index = self.indexes_per_class[class_id][pointer]

                # When finish one epoch of a sound class, then shuffle its
                # indexes and reset pointer.
                if self.pointers_of_classes[class_id] >= self.samples_num_per_class[class_id]:
                    self.pointers_of_classes[class_id] = 0
                    self.random_state.shuffle(self.indexes_per_class[class_id])

                batch_meta.append({
                    'hdf5_path': self.hdf5_paths[index],
                    'index_in_hdf5': self.indexes_in_hdf5[index],
                    'class_id': class_id})

            yield batch_meta

    def __len__(self) -> int:
        return self.steps_per_epoch


class DistributedSamplerWrapper:
    def __init__(self, sampler: object) -> None:
        r"""Distributed wrapper of sampler.

        Args:
            sampler (Sampler object)

        Returns:
            None
        """

        self.sampler = sampler

    def __iter__(self) -> List:
        r"""Yield a part of mini-batch meta on each device.

        Args:
            None

        Returns:
            list_meta (List), a part of mini-batch meta.
        """

        if dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()

        else:
            num_replicas = 1
            rank = 0

        for list_meta in self.sampler:
            yield list_meta[rank:: num_replicas]

    def __len__(self) -> int:
        return len(self.sampler)
