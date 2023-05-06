from typing import Dict

import h5py
import numpy as np

from uss.utils import int16_to_float32


class Dataset:
    def __init__(self, steps_per_epoch) -> None:
        r"""This class takes the meta as input, and return the waveform, target
        and other information of the audio clip. This class is used by
        DataLoader.

        Args:
            steps_per_epoch (int): the number of steps in an epoch
        """
        self.steps_per_epoch = steps_per_epoch

    def __getitem__(self, meta) -> Dict:
        """Load waveform, target and other information of an audio clip.

        Args:
            meta (Dict): {
                "hdf5_path": str,
                "index_in_hdf5": int,
                "class_id": int,
            }

        Returns:
            data_dict (Dict): {
                "hdf5_path": str,
                "index_in_hdf5": int,
                "audio_name": str,
                "waveform": (clip_samples,),
                "target": (classes_num,),
                "class_id": int,
            }
        """

        hdf5_path = meta["hdf5_path"]
        index_in_hdf5 = meta["index_in_hdf5"]
        class_id = meta["class_id"]

        with h5py.File(hdf5_path, 'r') as hf:

            audio_name = hf["audio_name"][index_in_hdf5].decode()

            waveform = int16_to_float32(hf["waveform"][index_in_hdf5])
            waveform = waveform
            # shape: (clip_samples,)

            target = hf["target"][index_in_hdf5].astype(np.float32)
            # shape: (classes_num,)

        data_dict = {
            "hdf5_path": hdf5_path,
            "index_in_hdf5": index_in_hdf5,
            "audio_name": audio_name,
            "waveform": waveform,
            "target": target,
            "class_id": class_id,
        }

        return data_dict

    def __len__(self) -> int:
        return self.steps_per_epoch
