import argparse
import logging
import os
import pathlib
from torch.utils.data import BatchSampler
from typing import List, NoReturn
import pytorch_lightning as pl

from casa.data.data_modules import DataModule, Dataset
from casa.utils import create_logging, read_yaml, load_pretrained_model #, load_pretrained_sed_model, load_pretrained_at_model
from casa.data.samplers import BalancedSampler
from casa.data.data_modules import DataModule
from casa.models.resunet import *
from casa.losses import get_loss_function
from casa.models.pl_modules import LitModel
from casa.data.anchor_segment_detectors import AnchorSegmentDetector
from casa.data.anchor_segment_mixers import AnchorSegmentMixer
from casa.data.query_condition_extractors import QueryConditionExtractor

# import pytorch_lightning as pl
# from pytorch_lightning.plugins import DDPPlugin

# from audioset_source_separation.callbacks.audioset_callbacks import \
#     get_audioset_callbacks
# from audioset_source_separation.config import CLIP_SAMPLES
# from audioset_source_separation.data.batch_data_preprocessors import \
#     AudioSetBatchDataPreprocessor
# from audioset_source_separation.data.data_modules import DataModule, Dataset
# from audioset_source_separation.data.samplers import BalancedSampler
# from audioset_source_separation.losses import get_loss_function
# # from audioset_source_separation.models.cond_resunet_subbandtime import \
# #     CondResUNetSubbandTime
# # from audioset_source_separation.models.cond_unet import CondUNet
# # from audioset_source_separation.models.cond_unet_subbandtime import \
# #     CondUNetSubbandTime
# from audioset_source_separation.models.resunet import ResUNet30
# from audioset_source_separation.models.lightning_modules import \
#     LitSourceSeparation
# from audioset_source_separation.optimizers.lr_schedulers import get_lr_lambda
# from audioset_source_separation.utils import (create_logging,
#                                               load_pretrained_at_model,
#                                               load_pretrained_sed_model,
#                                               read_yaml)


def get_dirs(workspace: str, filename: str, config_yaml: str, devices_num: int) -> List[str]:
    r"""Get directories.

    Args:
        workspace: str
        filenmae: str
        config_yaml: str
        gpus: int, e.g., 0 for cpu and 8 for training with 8 gpu cards

    Returns:
        checkpoints_dir: str
        logs_dir: str
        logger: pl.loggers.TensorBoardLogger
        statistics_path: str
    """

    # save checkpoints dir
    checkpoints_dir = os.path.join(
        workspace,
        "checkpoints",
        filename,
        "config={},devices={}".format(pathlib.Path(config_yaml).stem, devices_num),
    )
    os.makedirs(checkpoints_dir, exist_ok=True)

    # logs dir
    logs_dir = os.path.join(
        workspace,
        "logs",
        filename,
        "config={},devices={}".format(pathlib.Path(config_yaml).stem, devices_num),
    )
    os.makedirs(logs_dir, exist_ok=True)

    # loggings
    create_logging(logs_dir, filemode="w")
    logging.info(args)

    # tb_logs_dir = os.path.join(workspace, "tensorboard_logs")
    # os.makedirs(tb_logs_dir, exist_ok=True)

    # experiment_name = os.path.join(filename, pathlib.Path(config_yaml).stem)
    # logger = pl.loggers.TensorBoardLogger(save_dir=tb_logs_dir, name=experiment_name)

    # statistics path
    statistics_path = os.path.join(
        workspace,
        "statistics",
        filename,
        "config={},devices={}".format(pathlib.Path(config_yaml).stem, devices_num),
        "statistics.pkl",
    )
    os.makedirs(os.path.dirname(statistics_path), exist_ok=True)

    return checkpoints_dir, logs_dir, statistics_path

'''
from torch.utils.data.sampler import Sampler
from typing import Dict, List, Union, Iterable, Iterator
class BatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            print(self.sampler)
            sampler_iter = iter(self.sampler)
            while True:
                print(self.sampler)
                batch = [next(sampler_iter) for _ in range(self.batch_size)]
                print(batch)
                from IPython import embed; embed(using=False); os._exit(0)
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
'''

def get_data_module(
    workspace: str, config_yaml: str, num_workers: int, devices_num: int,
) -> DataModule:
    r"""Create data_module. Mini-batch data can be obtained by:

    code-block:: python

        data_module.setup()

        for batch_data_dict in data_module.train_dataloader():
            print(batch_data_dict.keys())
            break

    Args:
        workspace: str
        config_yaml: str
        num_workers: int, e.g., 0 for non-parallel and 8 for using cpu cores
            for preparing data in parallel
        distributed: bool

    Returns:
        data_module: DataModule
    """

    # read configurations
    configs = read_yaml(config_yaml)
    indexes_hdf5_path = os.path.join(workspace, configs['data']['indexes_dict'])
    batch_size = configs['train']['batch_size_per_device'] * devices_num
    steps_per_epoch = configs['train']['steps_per_epoch']

    # dataset
    train_dataset = Dataset()

    # sampler
    train_sampler = BalancedSampler(
        indexes_hdf5_path=indexes_hdf5_path,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
    )

    # Sampler3()

    # from casa.data.samplers import Sampler3
    # train_sampler = BatchSampler(
    #     sampler=Sampler3(),
    #     batch_size=16, 
    #     drop_last=True,
    # )

    # data module
    data_module = DataModule(
        train_sampler=train_sampler,
        train_dataset=train_dataset,
        num_workers=num_workers,
    )

    # data_module.setup()
    # for batch_data_dict in data_module.train_dataloader():
    #     # print(batch_data_dict.keys())
    #     # batch_data_dict['audio_name']
    #     from IPython import embed; embed(using=False); os._exit(0)

    return data_module

def get_devices_num():
    
    devices_str = os.getenv("CUDA_VISIBLE_DEVICES")

    if not devices_str:
        raise Exception("Must set the CUDA_VISIBLE_DEVICES flag.")

    devices_num = len(devices_str.split(','))

    return devices_num


def train(args) -> NoReturn:
    r"""Train, evaluate, and save checkpoints.

    Args:
        workspace: str, directory of workspace
        gpus: int, number of GPUs to train
        config_yaml: str
    """

    # arguments & parameters
    workspace = args.workspace
    config_yaml = args.config_yaml
    filename = args.filename

    devices_num = get_devices_num()
    print(devices_num)

    # distributed = True if gpus > 1 else False
    # evaluate_device = "cuda"  # Evaluate on a single GPU card.

    # Read config file.
    configs = read_yaml(config_yaml)

    # sed_checkpoint_path = configs['sed_checkpoint_path']
    # at_checkpoint_path = configs['at_checkpoint_path']

    num_workers = configs['train']['num_workers']
    model_type = configs['model']['model_type']
    input_channels = configs['model']['input_channels']
    output_channels = configs['model']['output_channels']
    condition_size = configs['data']['condition_size']
    loss_type = configs['train']['loss_type']
    learning_rate = float(configs['train']['learning_rate'])

    sample_rate = configs['data']['sample_rate']


    # sed_checkpoint_path = configs["train"]["sed_checkpoint_path"]
    # at_checkpoint_path = configs["train"]["at_checkpoint_path"]
    # sample_rate = configs["train"]["sample_rate"]
    # input_channels = configs["train"]["input_channels"]
    # output_channels = configs["train"]["output_channels"]
    # frames_per_second = configs["train"]["frames_per_second"]
    # segment_seconds = configs["train"]["segment_seconds"]
    # condition_type = configs["train"]["condition_type"]
    # condition_size = configs["train"]["condition_size"]
    # is_gamma = configs["train"]["condition_settings"]["gamma"]
    # augmentation = configs["train"]["augmentation"]
    # model_type = configs["train"]["model_type"]
    # loss_type = configs["train"]["loss_type"]
    # learning_rate = float(configs["train"]["learning_rate"])
    # precision = configs["train"]["precision"]
    # early_stop_steps = configs["train"]["early_stop_steps"]
    # warm_up_steps = configs["train"]["warm_up_steps"]
    # reduce_lr_steps = configs["train"]["reduce_lr_steps"]

    # clip_samples = CLIP_SAMPLES

    # # paths
    checkpoints_dir, logs_dir, statistics_path = get_dirs(
        workspace, filename, config_yaml, devices_num,
    )

    # # Load pretrained sound event detection and audio tagging model.
    sed_model = load_pretrained_model(
        model_name=configs['sound_event_detection']['model_name'],
        checkpoint_path=configs['sound_event_detection']['checkpoint_path'],
        freeze=configs['sound_event_detection']['freeze'],
    )

    at_model = load_pretrained_model(
        model_name=configs['audio_tagging']['model_name'],
        checkpoint_path=configs['audio_tagging']['checkpoint_path'],
        freeze=configs['audio_tagging']['freeze'],
    )

    # data module
    data_module = get_data_module(
        workspace=workspace,
        config_yaml=config_yaml,
        num_workers=num_workers,
        devices_num=devices_num,
        # distributed=distributed,
    )

    # # data preprocessor
    # batch_data_preprocessor = AudioSetBatchDataPreprocessor(
    #     augmentation=augmentation,
    #     sed_model=sed_model,
    #     at_model=at_model,
    #     segment_seconds=segment_seconds,
    #     frames_per_second=frames_per_second,
    #     sample_rate=sample_rate,
    #     clip_samples=clip_samples,
    #     condition_type=condition_type,
    # )
    
    # model
    Model = eval(model_type)
    # Model = str_to_class(model_type)

    ss_model = Model(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
    )

    # # loss function
    loss_function = get_loss_function(loss_type)

    # # callbacks
    # callbacks = get_audioset_callbacks(
    #     config_yaml=config_yaml,
    #     workspace=workspace,
    #     checkpoints_dir=checkpoints_dir,
    #     statistics_path=statistics_path,
    #     logger=logger,
    #     ss_model=ss_model,
    #     at_model=at_model,
    #     evaluate_device=evaluate_device,
    # )
    # # callbacks = []

    # learning rate reduce function
    # lr_lambda = lambda step: get_lr_lambda(
    #     step, warm_up_steps=warm_up_steps, reduce_lr_steps=reduce_lr_steps
    # )

    anchor_segment_detector = AnchorSegmentDetector(
        sed_model=sed_model,
        clip_seconds=10.,
        segment_seconds=2.,
        frames_per_second=100,
        sample_rate=sample_rate,
    )

    anchor_segment_mixer = AnchorSegmentMixer(
        mix_num=2,
    )

    # AvoidConflictInBatch()

    # DataAugmentor()

    query_condition_extractor = QueryConditionExtractor(
        at_model=at_model,
        condition_type='embedding',
    )

    # pytorch-lightning model
    pl_model = LitModel(
        sed_model=sed_model,
        at_model=at_model,
        ss_model=ss_model,
        anchor_segment_detector=anchor_segment_detector,
        anchor_segment_mixer=anchor_segment_mixer,
        query_condition_extractor=query_condition_extractor,
        # batch_data_preprocessor=batch_data_preprocessor,
        loss_function=loss_function,
        learning_rate=learning_rate,
        # lr_lambda=lr_lambda,
    )

    # trainer
    # trainer = pl.Trainer(
    #     checkpoint_callback=False,
    #     gpus=gpus,
    #     callbacks=callbacks,
    #     max_steps=early_stop_steps,
    #     accelerator="ddp",
    #     sync_batchnorm=True,
    #     precision=precision,
    #     profiler="simple",
    #     replace_sampler_ddp=False,
    #     plugins=[DDPPlugin(find_unused_parameters=True)],
    # )
    trainer = pl.Trainer(
        devices='auto',
        num_nodes=1,
        callbacks=None,
        fast_dev_run=False,
        max_epochs=-1,
        use_distributed_sampler=False,
    )

    # Fit, evaluate, and save checkpoints.
    trainer.fit(
        model=pl_model, 
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule=data_module,
        ckpt_path=None,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )

    args = parser.parse_args()
    args.filename = pathlib.Path(__file__).stem

    train(args)