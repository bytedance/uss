import numpy as np
import os
import pickle
import pathlib

from casa.config import IX_TO_LB


def add():

    config_yaml = "01b"
    classes_num = 527
    final_sdris = []

    # for step in [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000]:
    for step in [20000, 40000, 60000, 80000, 100000]:

        stat_path = os.path.join("stats", pathlib.Path(config_yaml).stem, "step={}.pkl".format(step))

        stats_dict = pickle.load(open(stat_path, 'rb'))

        mean_sdris = {}

        for class_id in range(classes_num):
            mean_sdris[class_id] = np.nanmean(stats_dict['sdris_dict'][class_id])
            
        final_sdri = np.nanmean([mean_sdris[class_id] for class_id in range(classes_num)])
        final_sdris.append(final_sdri)

    print(final_sdris)

    # from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':

    add()