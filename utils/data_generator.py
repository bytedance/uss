import os
import sys
import numpy as np
import h5py
import csv
import time
import logging

from utilities import int16_to_float32


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
            self.hdf5_names = [hdf5_name.decode() for hdf5_name in hf['hdf5_name'][:]]
            self.indexes_in_part_hdf5 = hf['index_in_hdf5'][:]
       
        logging.info('Audio samples: {}'.format(len(self.audio_names)))
 
    def get_relative_hdf5_path(self, hdf5_name):
        if hdf5_name in ['balanced_train.h5', 'eval.h5']:
            return hdf5_name
        elif 'unbalanced_train' in hdf5_name:
            relative_path = os.path.join('unbalanced_train', hdf5_name)
        else:
            raise Exception('Incorrect hdf5_name!')

        return relative_path
    
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
            batch_index_class_pairs = []
            i = 0
            while i < batch_size:
                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)

                class_id = self.queue.pop(0)
                pointer = self.pointers_of_classes[class_id]
                self.pointers_of_classes[class_id] += 1
                audio_index = self.indexes_per_class[class_id][pointer]
                
                if self.pointers_of_classes[class_id] >= self.samples_num_per_class[class_id]:
                    self.pointers_of_classes[class_id] = 0
                    self.random_state.shuffle(self.indexes_per_class[class_id])

                # batch_indexes.append(audio_index)
                # batch_classes.append(audio_index)
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


def collect_fn(list_data_dict):
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    return np_data_dict
