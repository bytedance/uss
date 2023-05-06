import numpy as np
import os
import pickle
import pathlib
import matplotlib.pyplot as plt

from uss.config import IX_TO_LB


def add():

    config_yaml = "02a"
    classes_num = 527
    final_sdris = []

    for step in range(180000, 300001, 20000):
    # for step in [20000, 40000, 60000, 80000, 100000]:

        stat_path = os.path.join("stats", pathlib.Path(config_yaml).stem, "step={}.pkl".format(step))

        stats_dict = pickle.load(open(stat_path, 'rb'))

        mean_sdris = {}

        for class_id in range(classes_num):
            mean_sdris[class_id] = np.nanmean(stats_dict['sdris_dict'][class_id])
            
        final_sdri = np.nanmean([mean_sdris[class_id] for class_id in range(classes_num)])
        final_sdris.append(final_sdri)

    print(final_sdris)

    # from IPython import embed; embed(using=False); os._exit(0)

def add2():

    # stat_path = "./workspaces/uss/statistics/train/config=ss_model=resunet30,querynet=at_soft_adapt,gpus=1,devices=1/statistics.pkl"

    # stat_path = "./workspaces/uss/statistics/train/config=ss_model=resunet30,querynet=at_soft,gpus=1,devices=1/statistics.pkl"

    # stat_path = "/home/tiger/test9/statistics.pkl"

    stat_path = "/home/tiger/workspaces/uss/statistics/train/ss_model=resunet30,querynet=at_soft,data=full,devices=8/statistics.pkl"

    stats_dict = pickle.load(open(stat_path, 'rb'))

    sdris = []

    for i, stats in enumerate(stats_dict['test']):
        sdri = np.nanmean(list(stats["sdri_dict"].values()))
        sdris.append(sdri)
        # sdri = np.mean(list(stats.values()))

    plt.plot(sdris)
    plt.savefig('_zz.pdf')

    print(sdris)

def add3():
    pass


if __name__ == '__main__':

    add2()