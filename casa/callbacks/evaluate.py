import os
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only

from casa.evaluate import AudioSetEvaluator
from casa.utils import StatisticsContainer
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import logging



class EvaluateCallback(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        # evaluator,
        pl_model,
        balanced_train_eval_dir,
        test_eval_dir,
        classes_num,
        max_eval_per_class,
        evaluate_step_frequency,
        summary_writer,
        statistics_path,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        # self.evaluator = evaluator

        self.balanced_train_evaluator = AudioSetEvaluator(
            # pl_model=pl_model, 
            audios_dir=balanced_train_eval_dir, 
            classes_num=classes_num, 
            max_eval_per_class=max_eval_per_class,
        )

        self.test_evaluator = AudioSetEvaluator(
            # pl_model=pl_model, 
            audios_dir=test_eval_dir, 
            classes_num=classes_num, 
            max_eval_per_class=max_eval_per_class,
        )

        self.pl_model = pl_model

        self.evaluate_step_frequency = evaluate_step_frequency

        self.summary_writer = summary_writer

        self.statistics_container = StatisticsContainer(statistics_path)

    @rank_zero_only
    def on_train_batch_end(self, *args, **kwargs):
        """ Check if we should save a checkpoint after every train batch """
        trainer = args[0]
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if global_step == 1 or global_step % self.evaluate_step_frequency == 0:

            for split, evaluator in zip(["balanced_train", "test"], [self.balanced_train_evaluator, self.test_evaluator]):

                stats_dict = evaluator(pl_model=self.pl_model)

                median_sdris_dict = AudioSetEvaluator.get_median_metrics(
                    stats_dict=stats_dict, 
                    metric_type="sdris_dict",
                )

                sdri = np.nanmean(list(median_sdris_dict.values()))

                self.summary_writer.add_scalar("SDRi/{}".format(split), global_step=global_step, scalar_value=sdri)

                logging.info("    Flush tensorboard logs to {}".format(self.summary_writer.log_dir))

                self.statistics_container.append(
                    steps=global_step, 
                    statistics={"sdri_dict": median_sdris_dict}, 
                    split=split,
                    flush=True,
                )            
