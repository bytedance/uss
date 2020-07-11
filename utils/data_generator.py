import os
import sys
import numpy as np
import h5py
import csv
import time
import logging

from utilities import int16_to_float32


class SsAudioSetDataset(object):
    def __init__(self, clip_samples, classes_num):
        """This class takes the meta of an audio clip as input, and return 
        the waveform and target of the audio clip. This class is used by DataLoader. 

        Args:
          clip_samples: int
          classes_num: int
        """
        self.clip_samples = clip_samples
        self.classes_num = classes_num
    
    def __getitem__(self, meta):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'audio_name': str, 
            'hdf5_path': str, 
            'index_in_hdf5': int}

        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """
        if meta is None:
            """Dummy waveform and target. This is used for samples with mixup 
            lamda of 0."""
            audio_name = None
            waveform = np.zeros((self.clip_samples,), dtype=np.float32)
            target = np.zeros((self.classes_num,), dtype=np.float32)
        else:
            hdf5_path = meta['hdf5_path']
            index_in_hdf5 = meta['index_in_hdf5']

            with h5py.File(hdf5_path, 'r') as hf:
                audio_name = hf['audio_name'][index_in_hdf5].decode()
                waveform = int16_to_float32(hf['waveform'][index_in_hdf5])
                target = hf['target'][index_in_hdf5].astype(np.float32)


        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': target, 'class_id': meta['class_id']}
            
        return data_dict


class Base(object):
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



class SsBalancedSampler(Base):
    def __init__(self, indexes_hdf5_path, batch_size, black_list_csv=None, 
        random_seed=1234):
        """Balanced sampler. Generate batch meta for training. Data are equally 
        sampled from different sound classes.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        super(SsBalancedSampler, self).__init__(indexes_hdf5_path, 
            batch_size, black_list_csv, random_seed)
        
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
                        'audio_name': self.audio_names[index], 
                        'hdf5_path': self.hdf5_paths[index], 
                        'index_in_hdf5': self.indexes_in_hdf5[index], 
                        'target': self.targets[index], 
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



'''
class SsAudioSetDataset(object):
    def __init__(self, target_hdf5_path, waveform_hdf5s_dir, audio_length, classes_num):
        """AduioSet dataset for later used by DataLoader. This class takes an 
        audio index as input and output corresponding waveform and target. 
        """
        self.waveform_hdf5s_dir = waveform_hdf5s_dir
        self.audio_length = audio_length
        self.classes_num = classes_num

        with h5py.File(target_hdf5_path, 'r') as hf:
            """
            {'audio_name': (audios_num,) e.g. ['YtwJdQzi7x7Q.wav', ...], 
             'waveform': (audios_num, audio_length), 
             'target': (audios_num, classes_num)}
             """
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.hdf5_path = [hdf5_path.decode() for hdf5_path in hf['hdf5_path'][:]]
            self.indexes_in_part_hdf5 = hf['index_in_hdf5'][:]
       
        logging.info('Audio samples: {}'.format(len(self.audio_names)))
 

    def __getitem__(self, index_class_pair):
        """Load waveform and target of the audio index. If index is -1 then 
            return None. 
        
        Returns: {'audio_name': str, 'waveform': (audio_length,), 'target': (classes_num,)}
        """
        (index, class_id) = index_class_pair

        audio_name = self.audio_names[index]
        hdf5_name = self.hdf5_names[index]
        index_in_part_hdf5 = self.indexes_in_part_hdf5[index]

        relative_hdf5_path = self.get_relative_hdf5_path(hdf5_name)
        hdf5_path = os.path.join(self.waveform_hdf5s_dir, relative_hdf5_path)

        with h5py.File(hdf5_path, 'r') as hf:
            audio_name = hf['audio_name'][index_in_part_hdf5].decode()
            waveform = int16_to_float32(hf['waveform'][index_in_part_hdf5])
            target = hf['target'][index_in_part_hdf5].astype(np.float32)
                
        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': target, 'class_id': class_id}
            
        return data_dict
    
    def __len__(self):
        return len(self.audio_names)
'''
 
'''
class SsBalancedSampler(object):

    def __init__(self, target_hdf5_path, batch_size, random_seed=1234, verbose=1):
        """Balanced sampler. Generate audio indexes for DataLoader. 
        
        Args:
          target_hdf5_path: string
          black_list_csv: string
          batch_size: int
          start_mix_epoch: int, only do mix up after this samples have been 
            trained after this times. 
        """

        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        # Load target
        load_time = time.time()
        with h5py.File(target_hdf5_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.target = hf['target'][:].astype(np.float32)
        
        (self.audios_num, self.classes_num) = self.target.shape
        logging.info('Load target time: {:.3f} s'.format(time.time() - load_time))
        
        self.samples_num_per_class = np.sum(self.target, axis=0)
        logging.info('samples_num_per_class: {}'.format(
            self.samples_num_per_class.astype(np.int64)))
        
        self.indexes_per_class = []
        
        for k in range(self.classes_num):
            self.indexes_per_class.append(
                np.where(self.target[:, k] == 1)[0])
            
        # Shuffle indexes
        for k in range(self.classes_num):
            self.random_state.shuffle(self.indexes_per_class[k])
        
        self.queue = []
        self.pointers_of_classes = [0] * self.classes_num

    def expand_queue(self, queue):
        classes_set = np.arange(self.classes_num).tolist()
        self.random_state.shuffle(classes_set)
        queue += classes_set
        return queue

    def __iter__(self):
        """Generate audio indexes for training. 
        
        Returns: batch_indexes: (batch_size,). 
        """
        batch_size = self.batch_size

        while True:
            # batch_indexes = []
            # batch_classes = []
            batch_meta = []
            i = 0
            while i < batch_size:
                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)

                class_id = self.queue.pop(0)
                pointer = self.pointers_of_classes[class_id]
                self.pointers_of_classes[class_id] += 1
                # audio_index = self.indexes_per_class[class_id][pointer]

                
                if self.pointers_of_classes[class_id] >= self.samples_num_per_class[class_id]:
                    self.pointers_of_classes[class_id] = 0
                    self.random_state.shuffle(self.indexes_per_class[class_id])

                # batch_indexes.append(audio_index)
                # batch_classes.append(audio_index)
                batch_meta.append({'class_id': class_id})
                batch_index_class_pairs.append((audio_index, class_id))
                i += 1

            yield batch_index_class_pairs

    def __len__(self):
        return -1
        
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
'''

def collect_fn(list_data_dict):
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    return np_data_dict
