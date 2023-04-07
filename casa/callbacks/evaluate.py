import os
import lightning.pytorch as pl

from casa.evaluate import AudioSetEvaluator


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
            pl_model=pl_model, 
            audios_dir=balanced_train_eval_dir, 
            classes_num=classes_num, 
            max_eval_per_class=max_eval_per_class,
        )

        self.test_evaluator = AudioSetEvaluator(
            pl_model=pl_model, 
            audios_dir=test_eval_dir, 
            classes_num=classes_num, 
            max_eval_per_class=max_eval_per_class,
        )

        self.evaluate_step_frequency = evaluate_step_frequency

    def on_train_batch_end(self, *args, **kwargs):
        """ Check if we should save a checkpoint after every train batch """
        trainer = args[0]
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if global_step == 1 or global_step % self.evaluate_step_frequency == 0:
            
            balanced_train_stats_dict = self.balanced_train_evaluator()
            test_stats_dict = self.test_evaluator()
            # from IPython import embed; embed(using=False); os._exit(0)
