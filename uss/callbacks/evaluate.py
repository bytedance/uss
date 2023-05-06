import logging

import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.tensorboard import SummaryWriter

from uss.evaluate import AudioSetEvaluator
from uss.utils import StatisticsContainer, get_mean_sdr_from_dict


class EvaluateCallback(pl.Callback):
    def __init__(
        self,
        pl_model: pl.LightningModule,
        balanced_train_eval_dir: str,
        test_eval_dir: str,
        classes_num: int,
        max_eval_per_class: int,
        evaluate_step_frequency: int,
        summary_writer: SummaryWriter,
        statistics_path: str,
    ) -> None:
        """Evaluate on AudioSet separation.

        Args:
            pl_model (pl.LightningModule): universal source separation module
            balanced_train_eval_dir (str): directory of balanced train set for evaluation
            test_eval_dir (str): directory of test set for evaluation
            classes_num (int): sound classes number
            max_eval_per_class (int): the number of samples to evaluate for each sound class
            evaluate_step_frequency (int): evaluate every N steps
            summary_writer (SummaryWriter): used to write TensorBoard logs
            statistics_path (str): path to write statistics

        Returns:
            None
        """

        # Evaluators
        self.balanced_train_evaluator = AudioSetEvaluator(
            audios_dir=balanced_train_eval_dir,
            classes_num=classes_num,
            max_eval_per_class=max_eval_per_class,
        )

        self.test_evaluator = AudioSetEvaluator(
            audios_dir=test_eval_dir,
            classes_num=classes_num,
            max_eval_per_class=max_eval_per_class,
        )

        self.pl_model = pl_model

        self.evaluate_step_frequency = evaluate_step_frequency

        self.summary_writer = summary_writer

        # Statistics container
        self.statistics_container = StatisticsContainer(statistics_path)

    @rank_zero_only
    def on_train_batch_end(self, *args, **kwargs):
        r"""Evaluate every #evaluate_step_frequency steps."""

        trainer = args[0]
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if global_step == 1 or global_step % self.evaluate_step_frequency == 0:

            for split, evaluator in zip(["balanced_train", "test"], [
                                        self.balanced_train_evaluator, self.test_evaluator]):

                logging.info("------ {} ------".format(split))

                stats_dict = evaluator(pl_model=self.pl_model)

                median_sdris_dict = AudioSetEvaluator.get_median_metrics(
                    stats_dict=stats_dict,
                    metric_type="sdris_dict",
                )

                median_sdri = get_mean_sdr_from_dict(median_sdris_dict)
                logging.info("Average SDRi: {:.3f}".format(median_sdri))

                self.summary_writer.add_scalar(
                    "SDRi/{}".format(split),
                    global_step=global_step,
                    scalar_value=median_sdri)

                logging.info(
                    "    Flush tensorboard logs to {}".format(
                        self.summary_writer.log_dir))

                self.statistics_container.append(
                    steps=global_step,
                    statistics={"sdri_dict": median_sdris_dict},
                    split=split,
                    flush=True,
                )
