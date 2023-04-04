import os
import lightning.pytorch as pl


class Evaluate(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        evaluator,
        save_step_frequency,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.evaluator = evaluator
        self.save_step_frequency = save_step_frequency

    def on_train_batch_end(self, *args, **kwargs):
        """ Check if we should save a checkpoint after every train batch """
        trainer = args[0]
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if global_step == 1 or global_step % self.save_step_frequency == 0:
            
            self.evaluator()
            # from IPython import embed; embed(using=False); os._exit(0)


class Eva:
    def __init__(self, pl_model):
        self.pl_model = pl_model

    def __call__(self):

        import numpy as np
        import torch
        tmp = np.zeros((1, 1, 32000 * 2))
        tmp = torch.Tensor(tmp).to('cuda')

        cond = np.zeros((1, 2048))
        cond = torch.Tensor(cond).to('cuda')

        input_dict = {'mixture': tmp, 'condition': cond}

        output_dict = self.pl_model.ss_model(input_dict)
        sep_wav = output_dict['waveform'].data.cpu().numpy().squeeze()

        from IPython import embed; embed(using=False); os._exit(0)