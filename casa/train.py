import argparse
import logging
import os
import pathlib
from torch.utils.data import BatchSampler
from typing import List, NoReturn
import lightning.pytorch as pl
from torch.utils.tensorboard import SummaryWriter

from casa.data.datasets import Dataset
from casa.utils import create_logging, read_yaml, load_pretrained_model
from casa.models.query_nets import initialize_query_net
from casa.data.samplers import BalancedSampler
from casa.data.datamodules import DataModule
from casa.models.resunet import *
from casa.losses import get_loss_function
from casa.models.pl_modules import LitSeparation, get_model_class
from casa.data.anchor_segment_detectors import AnchorSegmentDetector
from casa.data.anchor_segment_mixers import AnchorSegmentMixer
from casa.data.query_condition_extractors import QueryConditionExtractor
from casa.callbacks.base import CheckpointEveryNSteps
from casa.callbacks.evaluate import EvaluateCallback
from casa.config import FRAMES_PER_SECOND, CLIP_SECONDS
from casa.optimizers.lr_schedulers import get_lr_lambda


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
    yaml_name = pathlib.Path(config_yaml).stem

    # save checkpoints dir
    checkpoints_dir = os.path.join(
        workspace,
        "checkpoints",
        filename,
        "config={},devices={}".format(yaml_name, devices_num),
    )
    os.makedirs(checkpoints_dir, exist_ok=True)

    # logs dir
    logs_dir = os.path.join(
        workspace,
        "logs",
        filename,
        "config={},devices={}".format(yaml_name, devices_num),
    )
    os.makedirs(logs_dir, exist_ok=True)

    # loggings
    create_logging(logs_dir, filemode="w")
    logging.info(args)

    tf_logs_dir = os.path.join(
        workspace,
        "tf_logs",
        filename,
        "config={},devices={}".format(yaml_name, devices_num),
    )

    # statistics path
    statistics_path = os.path.join(
        workspace,
        "statistics",
        filename,
        "config={},devices={}".format(yaml_name, devices_num),
        "statistics.pkl",
    )
    os.makedirs(os.path.dirname(statistics_path), exist_ok=True)

    return checkpoints_dir, logs_dir, tf_logs_dir, statistics_path
 

def get_datamodule(
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

    # Read config file.
    configs = read_yaml(config_yaml)

    clip_seconds = CLIP_SECONDS
    frames_per_second = FRAMES_PER_SECOND
    sample_rate = configs['data']['sample_rate']
    classes_num = configs['data']['classes_num']
    segment_seconds = configs['data']['segment_seconds']
    mix_num = configs['data']['mix_num']
    anchor_segment_detect_mode = configs["data"]["anchor_segment_detect_mode"]

    ss_model_type = configs['ss_model']['model_type']
    input_channels = configs['ss_model']['input_channels']
    output_channels = configs['ss_model']['output_channels']
    condition_size = configs['query_net']['outputs_num']
    # condition_size = configs['data']['condition_size']
    # condition_type = configs['data']['condition_type']

    num_workers = configs['train']['num_workers']
    loss_type = configs['train']['loss_type']
    optimizer_type = configs["train"]["optimizer"]["optimizer_type"]
    learning_rate = float(configs['train']["optimizer"]['learning_rate'])
    lr_lambda_type = configs['train']["optimizer"]['lr_lambda_type']
    warm_up_steps = configs['train']["optimizer"]['warm_up_steps']
    reduce_lr_steps = configs['train']["optimizer"]['reduce_lr_steps']

    save_step_frequency = configs['train']['save_step_frequency']
    evaluate_step_frequency = configs['train']['evaluate_step_frequency']

    balanced_train_eval_dir = os.path.join(workspace, configs["evaluate"]["balanced_train_eval_dir"])
    test_eval_dir = os.path.join(workspace, configs["evaluate"]["test_eval_dir"])
    max_eval_per_class = configs["evaluate"]["max_eval_per_class"]

    # # paths
    checkpoints_dir, logs_dir, tf_logs_dir, statistics_path = get_dirs(
        workspace, filename, config_yaml, devices_num,
    )

    # # Load pretrained sound event detection and audio tagging model.
    sed_model = load_pretrained_model(
        model_type=configs['sound_event_detection']['model_type'],
        checkpoint_path=configs['sound_event_detection']['checkpoint_path'],
        freeze=configs['sound_event_detection']['freeze'],
    )

    query_net = initialize_query_net(
        configs=configs,
    )

    # data module
    datamodule = get_datamodule(
        workspace=workspace,
        config_yaml=config_yaml,
        num_workers=num_workers,
        devices_num=devices_num,
    )
    
    # model
    SsModel = get_model_class(model_type=ss_model_type)

    ss_model = SsModel(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
    )

    # # loss function
    loss_function = get_loss_function(loss_type)

    anchor_segment_detector = AnchorSegmentDetector(
        sed_model=sed_model,
        clip_seconds=clip_seconds,
        segment_seconds=segment_seconds,
        frames_per_second=frames_per_second,
        sample_rate=sample_rate,
        detect_mode=anchor_segment_detect_mode,
    )

    anchor_segment_mixer = AnchorSegmentMixer(
        mix_num=mix_num,
    )

    '''
    query_condition_extractor = QueryConditionExtractor(
        query_net=query_net,
        condition_type=condition_type,
    )
    '''

    lr_lambda_func = get_lr_lambda(
        lr_lambda_type=lr_lambda_type,
        warm_up_steps=warm_up_steps,
        reduce_lr_steps=reduce_lr_steps,
    )

    # pytorch-lightning model
    pl_model = LitSeparation(
        ss_model=ss_model,
        anchor_segment_detector=anchor_segment_detector,
        anchor_segment_mixer=anchor_segment_mixer,
        query_net=query_net,
        loss_function=loss_function,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        lr_lambda_func=lr_lambda_func,
    )

    checkpoint_every_n_steps = CheckpointEveryNSteps(
        checkpoints_dir=checkpoints_dir,
        save_step_frequency=save_step_frequency,
    )

    summary_writer = SummaryWriter(log_dir=tf_logs_dir)

    evaluate_callback = EvaluateCallback(
        pl_model=pl_model,
        balanced_train_eval_dir=balanced_train_eval_dir,
        test_eval_dir=test_eval_dir,
        classes_num=classes_num,
        max_eval_per_class=max_eval_per_class,
        evaluate_step_frequency=evaluate_step_frequency,
        summary_writer=summary_writer,
        statistics_path=statistics_path,
    )

    # callbacks = [checkpoint_callback, checkpoint_callback2]
    # callbacks = []
    # callbacks = [checkpoint_every_n_steps]
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
        use_distributed_sampler=False,
        sync_batchnorm=True,
        num_sanity_val_steps=2,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        strategy='ddp_find_unused_parameters_true',
    )

    # Fit, evaluate, and save checkpoints.
    trainer.fit(
        model=pl_model, 
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule=datamodule,
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