import argparse
import logging
import os
import pathlib
from torch.utils.data import BatchSampler
from typing import List, NoReturn
import lightning.pytorch as pl

from casa.data.datamodules import DataModule
from casa.data.datasets import Dataset
from casa.utils import create_logging, read_yaml, load_pretrained_model #, load_pretrained_sed_model, load_pretrained_at_model
from casa.data.samplers import BalancedSampler
from casa.data.datamodules import DataModule
from casa.models.resunet import *
from casa.losses import get_loss_function
from casa.models.pl_modules import LitSeparation
from casa.data.anchor_segment_detectors import AnchorSegmentDetector
from casa.data.anchor_segment_mixers import AnchorSegmentMixer
from casa.data.query_condition_extractors import QueryConditionExtractor
from casa.callbacks.base import CheckpointEveryNSteps
from casa.callbacks.evaluate import Evaluate


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
    train_dataset = Dataset(
        steps_per_epoch=steps_per_epoch,
    )

    # sampler
    train_sampler = BalancedSampler(
        indexes_hdf5_path=indexes_hdf5_path,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
    )

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
    # from IPython import embed; embed(using=False); os._exit(0)

    return data_module

# def get_devices_num():
    
#     devices_str = os.getenv("CUDA_VISIBLE_DEVICES")

#     if not devices_str:
#         raise Exception("Must set the CUDA_VISIBLE_DEVICES flag.")

#     devices_num = len(devices_str.split(','))

#     return devices_num


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

    devices_num = torch.cuda.device_count()

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
    condition_type = configs['data']['condition_type']

    sample_rate = configs['data']['sample_rate']

    save_step_frequency = configs['train']['save_step_frequency']

    # steps_per_epoch = configs["train"]["steps_per_epoch"]


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
    pl_model = LitSeparation(
        ss_model=ss_model,
        anchor_segment_detector=anchor_segment_detector,
        anchor_segment_mixer=anchor_segment_mixer,
        query_condition_extractor=query_condition_extractor,
        loss_function=loss_function,
        learning_rate=learning_rate,
        lr_lambda=None,
    )

    from lightning.pytorch.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath="./tmp", 
        save_top_k=3, monitor="val_loss"
    )

    checkpoint_every_n_steps = CheckpointEveryNSteps(save_step_frequency=save_step_frequency)

    from casa.callbacks.evaluate import Eva
    evaluator = Eva(pl_model=pl_model)

    # aa()

    evaluate_callback = Evaluate(
        evaluator=evaluator,
        save_step_frequency=save_step_frequency,
    )

    # callbacks = [checkpoint_callback, checkpoint_callback2]
    # callbacks = []
    # callbacks = [checkpoint_every_n_steps, evaluate_callback]
    callbacks = [checkpoint_every_n_steps, evaluate_callback]

    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        num_nodes=1,
        precision="32-true",
        logger=None,
        callbacks=callbacks,
        fast_dev_run=False,
        max_epochs=-1,
        log_every_n_steps=50,
        use_distributed_sampler=False,
        sync_batchnorm=True,
        num_sanity_val_steps=2,
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