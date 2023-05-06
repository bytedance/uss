import argparse
import logging
import os
import pathlib
from typing import List

import lightning.pytorch as pl
import torch
from torch.utils.tensorboard import SummaryWriter

from uss.callbacks.base import CheckpointEveryNSteps
from uss.callbacks.evaluate import EvaluateCallback
from uss.config import CLIP_SECONDS, FRAMES_PER_SECOND, panns_paths_dict
from uss.data.anchor_segment_detectors import AnchorSegmentDetector
from uss.data.anchor_segment_mixers import AnchorSegmentMixer
from uss.data.datamodules import DataModule
from uss.data.datasets import Dataset
from uss.data.samplers import BalancedSampler
from uss.losses import get_loss_function
from uss.models.pl_modules import LitSeparation, get_model_class
from uss.models.query_nets import initialize_query_net
from uss.optimizers.lr_schedulers import get_lr_lambda
from uss.utils import (create_logging, get_path, load_pretrained_panns,
                       parse_yaml)


def train(args) -> None:
    r"""Train, evaluate, and save checkpoints.

    Args:
        workspace (str): directory of workspace
        config_yaml (str): config yaml path

    Returns:
        None
    """

    # Arguments & parameters
    workspace = args.workspace
    config_yaml = args.config_yaml
    filename = args.filename

    # GPUs number
    devices_num = torch.cuda.device_count()

    # Read config file
    configs = parse_yaml(config_yaml)

    # Configurations of pretrained sound event detection model from PANNs
    sed_model_type = configs["sound_event_detection"]["model_type"]

    # Configuration of data to train the universal source separation system
    clip_seconds = CLIP_SECONDS
    frames_per_second = FRAMES_PER_SECOND
    sample_rate = configs["data"]["sample_rate"]
    classes_num = configs["data"]["classes_num"]
    segment_seconds = configs["data"]["segment_seconds"]
    anchor_segment_detect_mode = configs["data"]["anchor_segment_detect_mode"]
    mix_num = configs["data"]["mix_num"]
    match_energy = configs["data"]["augmentation"]["match_energy"]

    # Configuration of the universal source separation model
    ss_model_type = configs["ss_model"]["model_type"]
    input_channels = configs["ss_model"]["input_channels"]
    output_channels = configs["ss_model"]["output_channels"]
    condition_size = configs["query_net"]["outputs_num"]

    # Configuration of the trainer
    num_workers = configs["train"]["num_workers"]
    loss_type = configs["train"]["loss_type"]
    optimizer_type = configs["train"]["optimizer"]["optimizer_type"]
    learning_rate = float(configs["train"]["optimizer"]["learning_rate"])
    lr_lambda_type = configs["train"]["optimizer"]["lr_lambda_type"]
    warm_up_steps = configs["train"]["optimizer"]["warm_up_steps"]
    reduce_lr_steps = configs["train"]["optimizer"]["reduce_lr_steps"]
    save_step_frequency = configs["train"]["save_step_frequency"]
    evaluate_step_frequency = configs["train"]["evaluate_step_frequency"]
    resume_checkpoint_path = configs["train"]["resume_checkpoint_path"]
    if resume_checkpoint_path == "":
        resume_checkpoint_path = None

    # Configuration of the evaluation
    balanced_train_eval_dir = os.path.join(
        workspace, configs["evaluate"]["balanced_train_eval_dir"])
    test_eval_dir = os.path.join(
        workspace, configs["evaluate"]["test_eval_dir"])
    max_eval_per_class = configs["evaluate"]["max_eval_per_class"]

    # Get directories and paths
    checkpoints_dir, logs_dir, tf_logs_dir, statistics_path = get_dirs(
        workspace, filename, config_yaml, devices_num,
    )

    # Create a PyTorch Lightning datamodule
    datamodule = get_datamodule(
        workspace=workspace,
        config_yaml=config_yaml,
        num_workers=num_workers,
        devices_num=devices_num,
    )

    sed_model = load_pretrained_panns(
        model_type=sed_model_type,
        checkpoint_path=get_path(panns_paths_dict[sed_model_type]),
        freeze=True,
    )

    # Initialize query net
    query_net = initialize_query_net(
        configs=configs,
    )

    # Initialize separation model
    SsModel = get_model_class(model_type=ss_model_type)

    ss_model = SsModel(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
    )

    # Loss function
    loss_function = get_loss_function(loss_type=loss_type)

    # Anchor segment detector
    anchor_segment_detector = AnchorSegmentDetector(
        sed_model=sed_model,
        clip_seconds=clip_seconds,
        segment_seconds=segment_seconds,
        frames_per_second=frames_per_second,
        sample_rate=sample_rate,
        detect_mode=anchor_segment_detect_mode,
    )

    # Anchor segment mixer
    anchor_segment_mixer = AnchorSegmentMixer(
        mix_num=mix_num,
        match_energy=match_energy,
    )

    # Learning rate scaler
    lr_lambda_func = get_lr_lambda(
        lr_lambda_type=lr_lambda_type,
        warm_up_steps=warm_up_steps,
        reduce_lr_steps=reduce_lr_steps,
    )

    # PyTorch Lightning model
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

    # Checkpoint
    checkpoint_every_n_steps = CheckpointEveryNSteps(
        checkpoints_dir=checkpoints_dir,
        save_step_frequency=save_step_frequency,
    )

    # Summary writer
    summary_writer = SummaryWriter(log_dir=tf_logs_dir)

    # Evaluation callback
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

    # All callbacks
    callbacks = [checkpoint_every_n_steps, evaluate_callback]

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
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
        strategy="ddp_find_unused_parameters_true",
    )

    # Fit, evaluate, and save checkpoints
    trainer.fit(
        model=pl_model,
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule=datamodule,
        ckpt_path=resume_checkpoint_path,
    )


def get_dirs(
    workspace: str,
    filename: str,
    config_yaml: str,
    devices_num: int
) -> List[str]:
    r"""Get directories and paths.

    Args:
        workspace (str): directory of workspace
        filename (str): filename of current .py file.
        config_yaml (str): config yaml path
        devices_num (int): 0 for cpu and 8 for training with 8 GPUs

    Returns:
        checkpoints_dir (str): directory to save checkpoints
        logs_dir (str), directory to save logs
        tf_logs_dir (str), directory to save TensorBoard logs
        statistics_path (str), directory to save statistics
    """

    yaml_name = pathlib.Path(config_yaml).stem

    # Directory to save checkpoints
    checkpoints_dir = os.path.join(
        workspace,
        "checkpoints",
        filename,
        "{},devices={}".format(yaml_name, devices_num),
    )
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Directory to save logs
    logs_dir = os.path.join(
        workspace,
        "logs",
        filename,
        "{},devices={}".format(yaml_name, devices_num),
    )
    os.makedirs(logs_dir, exist_ok=True)

    # Directory to save TensorBoard logs
    create_logging(logs_dir, filemode="w")
    logging.info(args)

    tf_logs_dir = os.path.join(
        workspace,
        "tf_logs",
        filename,
        "{},devices={}".format(yaml_name, devices_num),
    )

    # Directory to save statistics
    statistics_path = os.path.join(
        workspace,
        "statistics",
        filename,
        "{},devices={}".format(yaml_name, devices_num),
        "statistics.pkl",
    )
    os.makedirs(os.path.dirname(statistics_path), exist_ok=True)

    return checkpoints_dir, logs_dir, tf_logs_dir, statistics_path


def get_datamodule(
    workspace: str,
    config_yaml: str,
    num_workers: int,
    devices_num: int,
) -> DataModule:
    r"""Create a PyTorch Lightning datamodule for yielding mini-batches of data.

    Args:
        workspace (str): directory of workspace
        config_yaml (str): config yaml path
        num_workers (int): e.g., 16 for using multiple cpu cores for preparing
            data in parallel
        devices_num (int): the number of GPUs to run

    Returns:
        datamodule: DataModule

    Examples::

        >>> data_module.setup()
        >>> for batch_data_dict in datamodule:
        >>>     print(batch_data_dict.keys())
        >>>     break
    """

    # Read configs
    configs = parse_yaml(config_yaml)
    indexes_hdf5_path = os.path.join(
        workspace, configs["data"]["indexes_dict"])
    batch_size = configs["train"]["batch_size_per_device"] * devices_num
    steps_per_epoch = configs["train"]["steps_per_epoch"]

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

    return data_module


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
