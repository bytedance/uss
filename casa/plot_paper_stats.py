import os
import librosa
import numpy as np


def add():

	# dataset_dir = "/mnt/bd/kqq3/datasets/dcase2018/task2/dataset_root/FSDKaggle2018.audio_train"
	dataset_dir = "/mnt/bd/kqq3/datasets/dcase2018/task2/dataset_root/FSDKaggle2018.audio_test"

	audio_names = sorted(os.listdir(dataset_dir))

	durations = []

	for audio_name in audio_names:
		audio_path = os.path.join(dataset_dir, audio_name)
		duration = librosa.get_duration(filename=audio_path)
		durations.append(duration)

	np.sum(durations)
	from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':
	add()