import logging
import os
import pathlib
import pickle
from typing import Dict
import time

import numpy as np
import h5py
from pytorch_lightning.utilities import rank_zero_only
import torch.distributed as dist


class Base:
    def __init__(self, indexes_hdf5_path, batch_size, black_list_csv, random_seed):
        """Base class of train sampler.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        # Black list
        if black_list_csv:
            self.black_list_names = read_black_list(black_list_csv)
        else:
            self.black_list_names = []

        logging.info('Black list samples: {}'.format(len(self.black_list_names)))

        # Load target
        load_time = time.time()

        with h5py.File(indexes_hdf5_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.hdf5_paths = [hdf5_path.decode() for hdf5_path in hf['hdf5_path'][:]]
            self.indexes_in_hdf5 = hf['index_in_hdf5'][:]
            self.targets = hf['target'][:].astype(np.float32)
        
        (self.audios_num, self.classes_num) = self.targets.shape
        logging.info('Training number: {}'.format(self.audios_num))
        logging.info('Load target time: {:.3f} s'.format(time.time() - load_time))



class BalancedSampler(Base):
    def __init__(self, indexes_hdf5_path, batch_size, steps_per_epoch, black_list_csv=None, 
        random_seed=1234, drop_last=True):
        """Balanced sampler. Generate batch meta for training. Data are equally 
        sampled from different sound classes.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        super(BalancedSampler, self).__init__(indexes_hdf5_path, 
            batch_size, black_list_csv, random_seed)
        
        self.steps_per_epoch = steps_per_epoch

        self.samples_num_per_class = np.sum(self.targets, axis=0)
        logging.info('samples_num_per_class: {}'.format(
            self.samples_num_per_class.astype(np.int32)))
        
        # Training indexes of all sound classes. E.g.: 
        # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
        self.indexes_per_class = []
        
        for k in range(self.classes_num):
            self.indexes_per_class.append(
                np.where(self.targets[:, k] == 1)[0])
            
        # Shuffle indexes
        for k in range(self.classes_num):
            self.random_state.shuffle(self.indexes_per_class[k])
        
        self.queue = []
        self.pointers_of_classes = [0] * self.classes_num

        self.drop_last = drop_last
        self.steps_per_epoch = steps_per_epoch
        
    def expand_queue(self, queue):
        classes_set = np.arange(self.classes_num).tolist()
        self.random_state.shuffle(classes_set)
        queue += classes_set
        return queue

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'audio_name': 'YfWBzCRl6LUs.wav', 
             'hdf5_path': 'xx/balanced_train.h5', 
             'index_in_hdf5': 15734, 
             'target': [0, 1, 0, 0, ...]}, 
            ...]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)

                class_id = self.queue.pop(0)
                pointer = self.pointers_of_classes[class_id]
                self.pointers_of_classes[class_id] += 1
                index = self.indexes_per_class[class_id][pointer]
                
                # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
                if self.pointers_of_classes[class_id] >= self.samples_num_per_class[class_id]:
                    self.pointers_of_classes[class_id] = 0
                    self.random_state.shuffle(self.indexes_per_class[class_id])

                # If audio in black list then continue
                if self.audio_names[index] in self.black_list_names:
                    continue
                else:
                    batch_meta.append({
                        'hdf5_path': self.hdf5_paths[index], 
                        'index_in_hdf5': self.indexes_in_hdf5[index], 
                        'class_id': class_id})
                    i += 1

            yield batch_meta

    def state_dict(self):
        state = {
            'indexes_per_class': self.indexes_per_class, 
            'queue': self.queue, 
            'pointers_of_classes': self.pointers_of_classes}
        return state
            
    def load_state_dict(self, state):
        self.indexes_per_class = state['indexes_per_class']
        self.queue = state['queue']
        self.pointers_of_classes = state['pointers_of_classes']

    def __len__(self):
        return self.steps_per_epoch


class BalancedSampler2(Base):
    def __init__(self, indexes_hdf5_path, batch_size, steps_per_epoch, black_list_csv=None, 
        random_seed=1234):
        """Balanced sampler. Generate batch meta for training. Data are equally 
        sampled from different sound classes.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        super(BalancedSampler2, self).__init__(indexes_hdf5_path, 
            batch_size, black_list_csv, random_seed)
        
        self.steps_per_epoch = steps_per_epoch

        self.samples_num_per_class = np.sum(self.targets, axis=0)
        logging.info('samples_num_per_class: {}'.format(
            self.samples_num_per_class.astype(np.int32)))
        
        # Training indexes of all sound classes. E.g.: 
        # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
        self.indexes_per_class = []
        
        for k in range(self.classes_num):
            self.indexes_per_class.append(
                np.where(self.targets[:, k] == 1)[0])
            
        # Shuffle indexes
        # for k in range(self.classes_num):
        #     self.random_state.shuffle(self.indexes_per_class[k])
        # from IPython import embed; embed(using=False); os._exit(0)
        
        self.queue = []
        self.pointers_of_classes = [0] * self.classes_num

    def expand_queue(self, queue):
        while len(queue) < 132960:
            classes_set = np.arange(self.classes_num).tolist()
            self.random_state.shuffle(classes_set)
            [self.indexes_per_class[d][self.random_state.randint(0, len(self.indexes_per_class[d]) - 1)] for d in classes_set]
            queue += classes_set
        # from IPython import embed; embed(using=False); os._exit(0)
        return queue

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'audio_name': 'YfWBzCRl6LUs.wav', 
             'hdf5_path': 'xx/balanced_train.h5', 
             'index_in_hdf5': 15734, 
             'target': [0, 1, 0, 0, ...]}, 
            ...]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)

                class_id = self.queue.pop(0)
                # pointer = self.pointers_of_classes[class_id]
                # self.pointers_of_classes[class_id] += 1
                index = self.random_state.choice(self.indexes_per_class[class_id])
                
                # If audio in black list then continue
                if self.audio_names[index] in self.black_list_names:
                    continue
                else:
                    batch_meta.append({
                        'hdf5_path': self.hdf5_paths[index], 
                        'index_in_hdf5': self.indexes_in_hdf5[index], 
                        'class_id': class_id})
                    i += 1

            # from IPython import embed; embed(using=False); os._exit(0)
            yield batch_meta

    def state_dict(self):
        state = {
            'indexes_per_class': self.indexes_per_class, 
            'queue': self.queue, 
            'pointers_of_classes': self.pointers_of_classes}
        return state
            
    def load_state_dict(self, state):
        self.indexes_per_class = state['indexes_per_class']
        self.queue = state['queue']
        self.pointers_of_classes = state['pointers_of_classes']

    def __len__(self):
        return self.steps_per_epoch


class BalancedSampler3(Base):
    def __init__(self, indexes_hdf5_path, batch_size, steps_per_epoch, black_list_csv=None, 
        random_seed=1234):
        """Balanced sampler. Generate batch meta for training. Data are equally 
        sampled from different sound classes.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        super(BalancedSampler3, self).__init__(indexes_hdf5_path, 
            batch_size, black_list_csv, random_seed)
        
        self.steps_per_epoch = steps_per_epoch

        self.samples_num_per_class = np.sum(self.targets, axis=0)
        logging.info('samples_num_per_class: {}'.format(
            self.samples_num_per_class.astype(np.int32)))
        
        # Training indexes of all sound classes. E.g.: 
        # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
        self.indexes_per_class = []
        
        for k in range(self.classes_num):
            self.indexes_per_class.append(
                np.where(self.targets[:, k] == 1)[0])
            
        # Shuffle indexes
        # for k in range(self.classes_num):
        #     self.random_state.shuffle(self.indexes_per_class[k])
        # from IPython import embed; embed(using=False); os._exit(0)
        
        self.queue = []
        self.pointers_of_classes = [0] * self.classes_num

    # def expand_queue(self, queue):
    #     while len(queue) < 132960:
    #         classes_set = np.arange(self.classes_num).tolist()
    #         self.random_state.shuffle(classes_set)
    #         # [self.indexes_per_class[d][self.random_state.randint(0, len(self.indexes_per_class[d]) - 1)] for d in classes_set]
    #         queue += classes_set
    #     # from IPython import embed; embed(using=False); os._exit(0)
    #     return queue

    def expand_queue(self, queue):
        classes_set = np.arange(self.classes_num).tolist()
        self.random_state.shuffle(classes_set)
        queue += classes_set
        return queue

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'audio_name': 'YfWBzCRl6LUs.wav', 
             'hdf5_path': 'xx/balanced_train.h5', 
             'index_in_hdf5': 15734, 
             'target': [0, 1, 0, 0, ...]}, 
            ...]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)
                    # print(self.queue)

                class_id = self.queue.pop(0)
                # pointer = self.pointers_of_classes[class_id]
                # self.pointers_of_classes[class_id] += 1
                index = self.random_state.choice(self.indexes_per_class[class_id])
                
                # If audio in black list then continue
                if self.audio_names[index] in self.black_list_names:
                    continue
                else:
                    batch_meta.append({
                        'hdf5_path': self.hdf5_paths[index], 
                        'index_in_hdf5': self.indexes_in_hdf5[index], 
                        'class_id': class_id})
                    i += 1

            # from IPython import embed; embed(using=False); os._exit(0)
            yield batch_meta

    def state_dict(self):
        state = {
            'indexes_per_class': self.indexes_per_class, 
            'queue': self.queue, 
            'pointers_of_classes': self.pointers_of_classes}
        return state
            
    def load_state_dict(self, state):
        self.indexes_per_class = state['indexes_per_class']
        self.queue = state['queue']
        self.pointers_of_classes = state['pointers_of_classes']

    def __len__(self):
        return self.steps_per_epoch


class BalancedSampler4a(Base):
    def __init__(self, indexes_hdf5_path, batch_size, steps_per_epoch, black_list_csv=None, 
        random_seed=1234):
        """Balanced sampler. Generate batch meta for training. Data are equally 
        sampled from different sound classes.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        super(BalancedSampler4a, self).__init__(indexes_hdf5_path, 
            batch_size, black_list_csv, random_seed)
        
        self.steps_per_epoch = steps_per_epoch

        self.samples_num_per_class = np.sum(self.targets, axis=0)
        logging.info('samples_num_per_class: {}'.format(
            self.samples_num_per_class.astype(np.int32)))
        
        # Training indexes of all sound classes. E.g.: 
        # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
        self.indexes_per_class = []
        
        for k in range(self.classes_num):
            self.indexes_per_class.append(
                np.where(self.targets[:, k] == 1)[0])
            
        # Shuffle indexes
        for k in range(self.classes_num):
            self.random_state.shuffle(self.indexes_per_class[k])
        
        self.queue = []
        self.pointers_of_classes = [0] * self.classes_num

        
        # tmp = list(np.concatenate(self.indexes_per_class, axis=0))
        # for i in range(len(self.targets)):
        #     if i not in tmp:
        #         print(i)

        # from IPython import embed; embed(using=False); os._exit(0)  


    def expand_queue(self, queue):
        classes_set = np.arange(self.classes_num).tolist()
        self.random_state.shuffle(classes_set)
        queue += classes_set
        return queue

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'audio_name': 'YfWBzCRl6LUs.wav', 
             'hdf5_path': 'xx/balanced_train.h5', 
             'index_in_hdf5': 15734, 
             'target': [0, 1, 0, 0, ...]}, 
            ...]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)

                class_id = self.queue.pop(0)
                pointer = self.pointers_of_classes[class_id]
                self.pointers_of_classes[class_id] += 1
                index = self.indexes_per_class[class_id][pointer]
                
                # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
                if self.pointers_of_classes[class_id] >= self.samples_num_per_class[class_id]:
                    # from IPython import embed; embed(using=False); os._exit(0)
                    self.pointers_of_classes[class_id] = 0
                    self.random_state.shuffle(self.indexes_per_class[class_id])

                # If audio in black list then continue
                if self.audio_names[index] in self.black_list_names:
                    continue
                else:
                    batch_meta.append({
                        'hdf5_path': self.hdf5_paths[index], 
                        'index_in_hdf5': self.indexes_in_hdf5[index], 
                        'class_id': class_id})
                    i += 1
                # from IPython import embed; embed(using=False); os._exit(0)

            yield batch_meta

    def state_dict(self):
        state = {
            'indexes_per_class': self.indexes_per_class, 
            'queue': self.queue, 
            'pointers_of_classes': self.pointers_of_classes}
        return state
            
    def load_state_dict(self, state):
        self.indexes_per_class = state['indexes_per_class']
        self.queue = state['queue']
        self.pointers_of_classes = state['pointers_of_classes']

    def __len__(self):
        return self.steps_per_epoch


class BalancedSampler4b(Base):
    def __init__(self, indexes_hdf5_path, batch_size, steps_per_epoch, black_list_csv=None, 
        random_seed=1234):
        """Balanced sampler. Generate batch meta for training. Data are equally 
        sampled from different sound classes.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        super(BalancedSampler4b, self).__init__(indexes_hdf5_path, 
            batch_size, black_list_csv, random_seed)
        
        self.steps_per_epoch = steps_per_epoch

        self.samples_num_per_class = np.sum(self.targets, axis=0)
        logging.info('samples_num_per_class: {}'.format(
            self.samples_num_per_class.astype(np.int32)))
        
        # Training indexes of all sound classes. E.g.: 
        # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
        self.indexes_per_class = []
        
        for k in range(self.classes_num):
            self.indexes_per_class.append(
                np.where(self.targets[:, k] == 1)[0])
            
        # Shuffle indexes
        for k in range(self.classes_num):
            self.random_state.shuffle(self.indexes_per_class[k])
        
        self.queue = []
        self.pointers_of_classes = [0] * self.classes_num

        
        # tmp = list(np.concatenate(self.indexes_per_class, axis=0))
        # for i in range(len(self.targets)):
        #     if i not in tmp:
        #         print(i)

        # from IPython import embed; embed(using=False); os._exit(0)  


    def expand_queue(self, queue):
        classes_set = np.arange(self.classes_num).tolist()
        self.random_state.shuffle(classes_set)
        queue += classes_set
        return queue

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'audio_name': 'YfWBzCRl6LUs.wav', 
             'hdf5_path': 'xx/balanced_train.h5', 
             'index_in_hdf5': 15734, 
             'target': [0, 1, 0, 0, ...]}, 
            ...]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)

                class_id = self.queue.pop(0)
                # pointer = self.pointers_of_classes[class_id]
                # self.pointers_of_classes[class_id] += 1
                # index = self.indexes_per_class[class_id][pointer]
                index = self.random_state.choice(self.indexes_per_class[class_id])
                
                # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
                # if self.pointers_of_classes[class_id] >= self.samples_num_per_class[class_id]:
                #     self.pointers_of_classes[class_id] = 0
                #     self.random_state.shuffle(self.indexes_per_class[class_id])

                # If audio in black list then continue
                if self.audio_names[index] in self.black_list_names:
                    continue
                else:
                    batch_meta.append({
                        'hdf5_path': self.hdf5_paths[index], 
                        'index_in_hdf5': self.indexes_in_hdf5[index], 
                        'class_id': class_id})
                    i += 1

            yield batch_meta

    def state_dict(self):
        state = {
            'indexes_per_class': self.indexes_per_class, 
            'queue': self.queue, 
            'pointers_of_classes': self.pointers_of_classes}
        return state
            
    def load_state_dict(self, state):
        self.indexes_per_class = state['indexes_per_class']
        self.queue = state['queue']
        self.pointers_of_classes = state['pointers_of_classes']

    def __len__(self):
        return self.steps_per_epoch


class UnBalancedSampler(Base):
    def __init__(self, indexes_hdf5_path, batch_size, steps_per_epoch, black_list_csv=None, 
        random_seed=1234):
        """Balanced sampler. Generate batch meta for training. Data are equally 
        sampled from different sound classes.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        super(UnBalancedSampler, self).__init__(indexes_hdf5_path, 
            batch_size, black_list_csv, random_seed)

        self.steps_per_epoch = steps_per_epoch
        self.audios_num = self.targets.shape[0]

        self.indexes = np.arange(self.audios_num)
            
        # Shuffle indexes
        self.random_state.shuffle(self.indexes)
        
        self.pointer = 0

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'audio_name': 'YfWBzCRl6LUs.wav', 
             'hdf5_path': 'xx/balanced_train.h5', 
             'index_in_hdf5': 15734, 
             'target': [0, 1, 0, 0, ...]}, 
            ...]
        """

        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                index = self.indexes[self.pointer]
                self.pointer += 1

                # Shuffle indexes and reset pointer
                if self.pointer >= self.audios_num:
                    self.pointer = 0
                    self.random_state.shuffle(self.indexes)
                
                tmp = np.where(self.targets[index]==1)[0]

                if len(tmp) == 0:
                    continue

                class_id = self.random_state.choice(tmp)
                # from IPython import embed; embed(using=False); os._exit(0)

                # If audio in black list then continue
                if self.audio_names[index] in self.black_list_names:
                    continue
                else:
                    batch_meta.append({
                        'hdf5_path': self.hdf5_paths[index], 
                        'index_in_hdf5': self.indexes_in_hdf5[index],
                        'class_id': class_id,
                    })
                    i += 1

            yield batch_meta

    def __len__(self):
        return self.steps_per_epoch


class KechenSampler:
    # def __init__(self, index_path, idc, config, factor = 3, eval_mode = False):
    def __init__(self, batch_size):

        self.batch_size = batch_size

        index_path = '/home/tiger/workspaces/audioset_source_separation/hdf5s/indexes/balanced_train.h5'
        
        idc = np.load('/home/tiger/workspaces/kechen_zeroshot2/balanced_train_idc.npy', allow_pickle = True)

        import kechen_config as _config
        self.config = _config

        factor = 3
        eval_mode = False

        self.random_state = np.random.RandomState(1234) 
        self.index_path = index_path
        self.fp = h5py.File(index_path, "r")
        # self.config = config
        self.idc = idc
        self.factor = factor
        self.classes_num = self.config.classes_num
        self.eval_mode = eval_mode
        self.total_size = int(len(self.fp["audio_name"]) * self.factor)
        self.generate_queue()
        logging.info("total dataset size: %d" %(self.total_size))
        logging.info("class num: %d" %(self.classes_num))
        # from IPython import embed; embed(using=False); os._exit(0) 

    def generate_queue(self):
        self.queue = []      
        self.class_queue = []
        if self.config.debug:
            self.total_size = 1000
        if self.config.balanced_data:
            while len(self.queue) < self.total_size * 2:
                if self.eval_mode:
                    if len(self.config.eval_list) == 0:
                        class_set = [*range(self.classes_num)]
                    else:
                        class_set = self.config.eval_list[:]
                else:
                    class_set = [*range(self.classes_num)]
                    class_set = list(set(class_set) - set(self.config.eval_list))
                self.random_state.shuffle(class_set)
                
                self.queue += [self.idc[d][self.random_state.randint(0, len(self.idc[d]) - 1)] for d in class_set]
                self.class_queue += class_set[:]

            # from IPython import embed; embed(using=False); os._exit(0)
            self.queue = self.queue[:self.total_size * 2]
            self.class_queue = self.class_queue[:self.total_size * 2]
            self.queue = [[self.queue[i],self.queue[i+1]] for i in range(0, self.total_size * 2, 2)]
            self.class_queue = [[self.class_queue[i],self.class_queue[i+1]] for i in range(0, self.total_size * 2, 2)]
            assert len(self.queue) == self.total_size, "generate data error!!"

        else:
            if self.eval_mode:
                    if len(self.config.eval_list) == 0:
                        class_set = [*range(self.classes_num)]
                    else:
                        class_set = self.config.eval_list[:]
            else:
                class_set = [*range(self.classes_num)]
                class_set = list(set(class_set) - set(self.config.eval_list))
            self.class_queue = self.random_state.choices(class_set, k = self.total_size * 2)
            self.queue = [self.idc[d][self.random_state.randint(0, len(self.idc[d]) - 1)] for d in self.class_queue]
            self.queue = [[self.queue[i],self.queue[i+1]] for i in range(0, self.total_size * 2, 2)]
            self.class_queue = [[self.class_queue[i],self.class_queue[i+1]] for i in range(0, self.total_size * 2, 2)]
            assert len(self.queue) == self.total_size, "generate data error!!" 
        logging.info("queue regenerated:%s" %(self.queue[-5:]))

        # from IPython import embed; embed(using=False); os._exit(0)
        self.pointer = 0

    def __iter__(self):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: {
            "audio_name_1": str,
            "waveform_1": (clip_samples,),
            "class_id_1": int,
            "audio_name_2": str,
            "waveform_2": (clip_samples,),
            "class_id_2": int,
            ...
            "check_num": int
        }
        """
        # put the right index here!!!
        '''
        meta = {}

        for k in range(2):
            s_index = self.queue[index][k]
            target = self.class_queue[index][k]
            audio_name = self.fp["audio_name"][s_index].decode()
            
            hdf5_path = self.fp["hdf5_path"][s_index].decode().replace("../workspace", self.config.dataset_path)
            index_in_hdf5 = self.fp["index_in_hdf5"][s_index]

            meta['hdf5_path_{}'.format(k)] = hdf5_path
            meta['index_in_hdf5_{}'.format(k)] = index_in_hdf5
            meta['class_id_{}'.format(k)] = target

        return meta
        '''

        while True:

            batch_meta = []

            while len(batch_meta) < self.batch_size * 2:

                if self.pointer == self.total_size:
                    self.pointer = 0
                    self.generate_queue()

                for k in range(2):
                    s_index = self.queue[self.pointer][k]
                    target = self.class_queue[self.pointer][k]
                    audio_name = self.fp["audio_name"][s_index].decode()
                    
                    hdf5_path = self.fp["hdf5_path"][s_index].decode().replace("../workspace", self.config.dataset_path)
                    index_in_hdf5 = self.fp["index_in_hdf5"][s_index]
                    # print(index_in_hdf5)

                    meta = {
                        'hdf5_path': hdf5_path,
                        'index_in_hdf5': index_in_hdf5,
                        'class_id': target
                    }
                    batch_meta.append(meta)

                self.pointer += 1
                # print(self.pointer) 

            # from IPython import embed; embed(using=False); os._exit(0)
            yield batch_meta


    def __len__(self):
        return self.total_size



class DistributedSamplerWrapper:
    def __init__(self, sampler):
        r"""Distributed wrapper of sampler.
        """

        # self.num_replicas = dist.get_world_size()
        # self.rank = dist.get_rank()
        self.sampler = sampler

    def __iter__(self):
        num_replicas = dist.get_world_size()
        rank = dist.get_rank()

        for indices in self.sampler:
            yield indices[rank :: num_replicas]

    def __len__(self):
        return len(self.sampler)
