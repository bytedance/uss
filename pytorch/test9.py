import os
import sys
import librosa

sys.path.insert(1, os.path.join(sys.path[0], '../pytorch'))
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
import _pickle as cPickle

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from sed_models import Cnn13_DecisionLevelMax
from ss_models import UNet
from utilities import (create_folder, get_filename, create_logging, StatisticsContainer)
import config
from losses import get_loss_func
from data_generator import SsAudioSetDataset, SsBalancedSampler, collect_fn 
from pytorch_utils import count_parameters, SedMix, debug_and_plot, move_data_to_device
from evaluate import Evaluator


def train(args):
    """Train AudioSet tagging model. 

    Args:
      dataset_dir: str
      workspace: str
      data_type: 'balanced_train' | 'unbalanced_train'
      frames_per_second: int
      mel_bins: int
      model_type: str
      loss_type: 'bce'
      balanced: bool
      augmentation: str
      batch_size: int
      learning_rate: float
      resume_iteration: int
      early_stop: int
      accumulation_steps: int
      cuda: bool
    """

    # Arugments & parameters
    workspace = args.workspace
    at_checkpoint_path = args.at_checkpoint_path
    data_type = args.data_type
    model_type = args.model_type
    condition_type = args.condition_type
    wiener_filter = args.wiener_filter
    loss_type = args.loss_type
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename

    num_workers = 8
    sample_rate = config.sample_rate
    audio_length = config.audio_length
    classes_num = config.classes_num
    loss_func = get_loss_func(loss_type)
    neighbour_segs = 2  # segments used for training has length of (neighbour_segs * 2 + 1) * 0.32 ~= 1.6 s
    eval_max_iteration = 2  # Number of mini_batches for validation
    

    # Paths
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'waveforms')

    eval_train_targets_hdf5_path = os.path.join(workspace, 
        'hdf5s', 'targets', 'balanced_train.h5')

    eval_test_targets_hdf5_path = os.path.join(workspace, 'hdf5s', 'targets', 
        'eval.h5')

    if data_type == 'balanced_train':
        train_targets_hdf5_path = os.path.join(workspace, 'hdf5s', 'targets', 
            'balanced_train.h5')
    elif data_type == 'full_train':
        train_targets_hdf5_path = os.path.join(workspace, 'hdf5s', 'targets', 
            'full_train.h5')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        'data_type={}'.format(data_type), model_type, 
        'condition_type={}'.format(condition_type), 'wiener_filter={}'.format(wiener_filter), 
        'loss_type={}'.format(loss_type), 'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)
    
    statistics_path = os.path.join(workspace, 'statistics', filename, 
        'data_type={}'.format(data_type), model_type, 
        'condition_type={}'.format(condition_type), 'wiener_filter={}'.format(wiener_filter), 
        'loss_type={}'.format(loss_type), 'batch_size={}'.format(batch_size), 
        'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, 
        'data_type={}'.format(data_type), model_type, 
        'condition_type={}'.format(condition_type), 'wiener_filter={}'.format(wiener_filter), 
        'loss_type={}'.format(loss_type), 'batch_size={}'.format(batch_size))

    create_logging(logs_dir, filemode='w')
    logging.info(args)
    
    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'

    # Dataset will be used by DataLoader later. Provide an index and return 
    # waveform and target of audio
    train_dataset = SsAudioSetDataset(
        target_hdf5_path=train_targets_hdf5_path, 
        waveform_hdf5s_dir=waveform_hdf5s_dir, 
        audio_length=audio_length, 
        classes_num=classes_num)

    bal_dataset = SsAudioSetDataset(
        target_hdf5_path=eval_train_targets_hdf5_path, 
        waveform_hdf5s_dir=waveform_hdf5s_dir, 
        audio_length=audio_length, 
        classes_num=classes_num)

    test_dataset = SsAudioSetDataset(
        target_hdf5_path=eval_test_targets_hdf5_path, 
        waveform_hdf5s_dir=waveform_hdf5s_dir, 
        audio_length=audio_length, 
        classes_num=classes_num)

    # Sampler
    train_sampler = SsBalancedSampler(
        target_hdf5_path=train_targets_hdf5_path, 
        batch_size=batch_size)
        
    bal_sampler = SsBalancedSampler(
        target_hdf5_path=eval_train_targets_hdf5_path, 
        batch_size=batch_size)

    test_sampler = SsBalancedSampler(
        target_hdf5_path=eval_test_targets_hdf5_path, 
        batch_size=batch_size)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
        batch_sampler=train_sampler, collate_fn=collect_fn, 
        num_workers=num_workers, pin_memory=True)

    bal_loader = torch.utils.data.DataLoader(dataset=bal_dataset, 
        batch_sampler=bal_sampler, collate_fn=collect_fn, 
        num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
        batch_sampler=test_sampler, collate_fn=collect_fn, 
        num_workers=num_workers, pin_memory=True)
    
    # Load pretrained SED model. Do not change these parameters as they are 
    # part of the pretrained SED model.
    sed_model = Cnn13_DecisionLevelMax(sample_rate=32000, window_size=1024, 
        hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)

    logging.info('Loading checkpoint {}'.format(at_checkpoint_path))
    checkpoint = torch.load(at_checkpoint_path)
    sed_model.load_state_dict(checkpoint['model'])
    sed_model = torch.nn.DataParallel(sed_model)

    if 'cuda' in str(device):
        sed_model.to(device)

    # Source separation model
    SsModel = eval(model_type)
    ss_model = SsModel(classes_num, condition_type, wiener_filter)
    
    params_num = count_parameters(ss_model)
    # flops_num = count_flops(model, audio_length)
    logging.info('Parameters num: {}'.format(params_num))
    # logging.info('Flops num: {:.3f} G'.format(flops_num / 1e9))
    
    # Resume training
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'batch_size={}'.format(batch_size), 
            '{}_iterations.pth'.format(resume_iteration))

        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        ss_model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = checkpoint['iteration']

    else:
        iteration = 0
    
    # Optimizer
    optimizer = optim.Adam(ss_model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    ss_model = torch.nn.DataParallel(ss_model)

    if 'cuda' in str(device):
        ss_model.to(device)
    
    # Statistics
    statistics_container = StatisticsContainer(statistics_path)
    
    sed_mix = SedMix(sed_model, neighbour_segs=neighbour_segs, sample_rate=sample_rate)

    # Evaluator
    bal_evaluator = Evaluator(
        generator=bal_loader, 
        sed_mix=sed_mix, 
        sed_model=sed_model, 
        ss_model=ss_model, 
        max_iteration=eval_max_iteration)

    test_evaluator = Evaluator(
        generator=bal_loader, 
        sed_mix=sed_mix, 
        sed_model=sed_model, 
        ss_model=ss_model, 
        max_iteration=eval_max_iteration)
     
    train_bgn_time = time.time()
    t1 = time.time()

    for batch_10s_dict in train_loader:
        '''
        # Evaluate 
        if (iteration % 2000 == 0 and iteration > resume_iteration) or (iteration == 0):
            train_fin_time = time.time()

            bal_statistics = bal_evaluator.evaluate()
            test_statistics = test_evaluator.evaluate()
            
            logging.info('Validate bal sdr: {:.3f}, sir: {:.3f}, sar: {:.3f}'.format(
                bal_statistics['sdr'], bal_statistics['sir'], bal_statistics['sar']))
            logging.info('Validate test sdr: {:.3f}, sir: {:.3f}, sar: {:.3f}'.format(
                test_statistics['sdr'], test_statistics['sir'], test_statistics['sar']))

            statistics_container.append(iteration, bal_statistics, data_type='bal')
            statistics_container.append(iteration, test_statistics, data_type='test')
            statistics_container.dump()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()
        
        # Save model
        if iteration % 20000 == 0:
            checkpoint = {
                'iteration': iteration, 
                'model': ss_model.module.state_dict(), 
                'optimizer': optimizer.state_dict(), 
                'sampler': train_sampler.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
        '''
        # Get mixture and target data
        batch_data_dict = sed_mix.get_mix_data(batch_10s_dict, sed_model, with_identity_zero=True)

        if False:
            debug_and_plot(ss_model, batch_10s_dict, batch_data_dict)
            
        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

        # Forward
        ss_model.train()
        batch_stft_output = ss_model(
            batch_data_dict['mixture'], 
            batch_data_dict['hard_condition'], 
            batch_data_dict['soft_condition'])

        # Target
        batch_stft_source = ss_model.module.wavin_to_target(
            batch_data_dict['source'])

        loss = loss_func(batch_stft_output, batch_stft_source)
        print(loss)

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
        if iteration % 10 == 0:
            print(iteration, 'time: {:.3f} s'.format(time.time() - t1))
            t1 = time.time()
        
        iteration += 1
        
        # Stop learning
        if iteration == early_stop:
            break



def inference(args):

    dataset_dir = args.dataset_dir
    workspace = args.workspace
    data_type = args.data_type
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    iteration = args.iteration
    accumulation_steps = args.accumulation_steps
    cuda = args.cuda and torch.cuda.is_available()
    filename = args.filename

    classes_num = config.classes_num
    sample_rate = config.sample_rate
    total_samples = sample_rate * config.clip_seconds
    total_frames = total_samples // hop_size
    num_workers = 0

    # Paths
    hdf5s_dir = os.path.join(dataset_dir, 'features', 'waveform_hdf5')
 
    black_list_csv = os.path.join(workspace, 'black_list', 'dcase2017task4.csv')
    
    class_labels_indices_path = os.path.join(dataset_dir, 'metadata', 
        'class_labels_indices.csv')

    test_targets_hdf5_path = os.path.join(dataset_dir, 'coarse_targets_hdf5', 
        'eval_target.h5')

    # test_targets_hdf5_path = os.path.join(dataset_dir, 'coarse_targets_hdf5', 
    #     'balanced_train_target.h5')

    if cuda:
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'
    
    # Load labels
    (ids, labels) = load_class_label_indices(class_labels_indices_path)
    classes_num = len(labels)
    
    # AT model
    sed_model = Cnn13_GMP_fromwaveform_no_scale_c_specaug_64x64(
        window_size=window_size, hop_size=hop_size, window=window, 
        center=center, pad_mode=pad_mode, sample_rate=sample_rate, 
        mel_bins=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, 
        top_db=top_db, freeze_parameters=True, scalar=None, 
        classes_num=classes_num, device=device)

    sed_model = nn.DataParallel(sed_model)

    if cuda:
        sed_model.cuda()

    at_checkpoint_path = '/mnt/cephfs_hl/speechsv/qiuqiang.kong/workspaces/audioset_tagging_cnn_transfer/checkpoints/main_segment/sample_rate=32000,window_size=1024,hop_size=500,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13_GMP_fromwaveform_no_scale_c_specaug_64x64/loss_type=bce/balanced=balanced/augmentation=none/batch_size=32/accumulation_steps=1/980000_iterations.pth'
        
    checkpoint = torch.load(at_checkpoint_path)
    sed_model.load_state_dict(checkpoint['model'])
    
    # SS Model
    Model = eval(model_type)
    model = Model(classes_num)
    
    params_num = count_parameters(model)
    logging.info('Parameters number: {}'.format(params_num))

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if cuda:
        model.cuda()

    ss_checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'accumulation_steps={}'.format(accumulation_steps), 
        '{}_iterations.pth'.format(iteration))
        
    checkpoint = torch.load(ss_checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    # Dataset, used by DataLoader later. Provide an index and return feature
    # of audio
    test_dataset = SSAudioSetDatasetBigHdf5(
        hdf5s_dir=hdf5s_dir, 
        target_hdf5_path=test_targets_hdf5_path)
    
    test_sampler = SSEvaluate2Sampler(target_hdf5_path=test_targets_hdf5_path, 
        black_list_csv=black_list_csv, batch_size=16)

    # Data loader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
        batch_sampler=test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    # Evaluator
    test_evaluator = EvaluatorHardSoft(
        model=model, 
        generator=test_loader, 
        sed_model=sed_model, 
        cuda=cuda, 
        write_wav=True)

    test_statistics = test_evaluator.evaluate()
    print(test_statistics)


def evaluate_all(args):

    dataset_dir = args.dataset_dir
    workspace = args.workspace
    workspace2 = args.workspace2
    data_type = args.data_type
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    iteration = args.iteration
    accumulation_steps = args.accumulation_steps
    cuda = args.cuda and torch.cuda.is_available()
    filename = args.filename

    classes_num = config.classes_num
    sample_rate = config.sample_rate
    total_samples = sample_rate * config.clip_seconds
    total_frames = total_samples // hop_size
    num_workers = 0

    # Paths
    hdf5s_dir = os.path.join(dataset_dir, 'features', 'waveform_hdf5')
 
    black_list_csv = os.path.join(workspace, 'black_list', 'dcase2017task4.csv')
    
    class_labels_indices_path = os.path.join(dataset_dir, 'metadata', 
        'class_labels_indices.csv')

    test_targets_hdf5_path = os.path.join(dataset_dir, 'coarse_targets_hdf5', 
        'eval_target.h5')

    # test_targets_hdf5_path = os.path.join(dataset_dir, 'coarse_targets_hdf5', 
    #     'balanced_train_target.h5')

    if cuda:
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'
    
    # Load labels
    (ids, labels) = load_class_label_indices(class_labels_indices_path)
    classes_num = len(labels)
    
    # AT model
    sed_model = Cnn13_GMP_fromwaveform_no_scale_c_specaug_64x64(
        window_size=window_size, hop_size=hop_size, window=window, 
        center=center, pad_mode=pad_mode, sample_rate=sample_rate, 
        mel_bins=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, 
        top_db=top_db, freeze_parameters=True, scalar=None, 
        classes_num=classes_num, device=device)

    sed_model = nn.DataParallel(sed_model)

    if cuda:
        sed_model.cuda()

    at_checkpoint_path = '/mnt/cephfs_hl/speechsv/qiuqiang.kong/workspaces/audioset_tagging_cnn_transfer/checkpoints/main_segment/sample_rate=32000,window_size=1024,hop_size=500,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13_GMP_fromwaveform_no_scale_c_specaug_64x64/loss_type=bce/balanced=balanced/augmentation=none/batch_size=32/accumulation_steps=1/980000_iterations.pth'
        
    checkpoint = torch.load(at_checkpoint_path)
    sed_model.load_state_dict(checkpoint['model'])
    
    # SS Model
    Model = eval(model_type)
    model = Model(classes_num)
    
    params_num = count_parameters(model)
    logging.info('Parameters number: {}'.format(params_num))

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if cuda:
        model.cuda()

    ss_checkpoint_path = os.path.join(workspace2, 'checkpoints', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'accumulation_steps={}'.format(accumulation_steps), 
        '{}_iterations.pth'.format(iteration))
 
    sdr_path = os.path.join(workspace2, 'sdrs', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'accumulation_steps={}'.format(accumulation_steps), '{}_sdr.pkl'.format(iteration))
    create_folder(os.path.dirname(sdr_path))
        
    checkpoint = torch.load(ss_checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    # Dataset, used by DataLoader later. Provide an index and return feature
    # of audio
    test_dataset = SSAudioSetDatasetBigHdf5(
        hdf5s_dir=hdf5s_dir, 
        target_hdf5_path=test_targets_hdf5_path)
    
    test_sampler = SSEvaluate2Sampler(target_hdf5_path=test_targets_hdf5_path, 
        black_list_csv=black_list_csv, batch_size=16)

    # Data loader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
        batch_sampler=test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    # Evaluator
    test_evaluator = EvaluatorHardSoft(
        model=model, 
        generator=test_loader, 
        sed_model=sed_model, 
        cuda=cuda, 
        write_wav=True)

    sdr_dict = test_evaluator.evaluate_all()
    print(sdr_dict)
 
    cPickle.dump(sdr_dict, open(sdr_path, 'wb'))
    print('Write to {}'.format(sdr_path))


def combine_sdrs(args):

    dataset_dir = args.dataset_dir
    workspace2 = args.workspace2
    data_type = args.data_type
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    bgn_iteration = args.bgn_iteration
    fin_iteration = args.fin_iteration
    accumulation_steps = args.accumulation_steps
    cuda = args.cuda and torch.cuda.is_available()
    filename = args.filename

    classes_num = config.classes_num
    sample_rate = config.sample_rate
    total_samples = sample_rate * config.clip_seconds
    total_frames = total_samples // hop_size
    num_workers = 0

    sdr_mat = []

    combined_sdr_path = os.path.join(workspace2, 'sdrs', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'accumulation_steps={}'.format(accumulation_steps), 'all_sdr.pkl')

    for iteration in range(bgn_iteration, fin_iteration, 10000):
        sdr_path = os.path.join(workspace2, 'sdrs', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            'accumulation_steps={}'.format(accumulation_steps), '{}_sdr.pkl'.format(iteration))

        sdr_dict = cPickle.load(open(sdr_path, 'rb'))

        sdrs = []
        for k in range(config.classes_num):
            sdr = np.mean(sdr_dict[k])
            sdrs.append(sdr)
        
        sdr_mat.append(sdrs)

    sdr_mat = np.array(sdr_mat)
    sorted_idxes = np.argsort(sdr_mat[0])[::-1]

    for idx in sorted_idxes:
        print('{}: {:.3f}'.format(config.lbs[idx], sdr_mat[0][idx]))

    cPickle.dump(sdr_mat, open(combined_sdr_path, 'wb'))
    print('Write to {}'.format(combined_sdr_path))

"""
def inference_new(args):

    dataset_dir = args.dataset_dir
    workspace = args.workspace
    data_type = args.data_type
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    iteration = args.iteration
    accumulation_steps = args.accumulation_steps
    cuda = args.cuda and torch.cuda.is_available()
    filename = args.filename

    classes_num = config.classes_num
    sample_rate = config.sample_rate
    total_samples = sample_rate * config.clip_seconds
    total_frames = total_samples // hop_size
    num_workers = 0

    # Paths
    hdf5s_dir = os.path.join(dataset_dir, 'features', 'waveform_hdf5')
 
    black_list_csv = os.path.join(workspace, 'black_list', 'dcase2017task4.csv')
    
    class_labels_indices_path = os.path.join(dataset_dir, 'metadata', 
        'class_labels_indices.csv')

    test_targets_hdf5_path = os.path.join(dataset_dir, 'coarse_targets_hdf5', 
        'eval_target.h5')

    # test_targets_hdf5_path = os.path.join(dataset_dir, 'coarse_targets_hdf5', 
    #     'balanced_train_target.h5')

    if cuda:
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'
    
    # Load labels
    (ids, labels) = load_class_label_indices(class_labels_indices_path)
    classes_num = len(labels)
    
    # AT model
    sed_model = Cnn13_GMP_fromwaveform_no_scale_c_specaug_64x64(
        window_size=window_size, hop_size=hop_size, window=window, 
        center=center, pad_mode=pad_mode, sample_rate=sample_rate, 
        mel_bins=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, 
        top_db=top_db, freeze_parameters=True, scalar=None, 
        classes_num=classes_num, device=device)

    sed_model = nn.DataParallel(sed_model)

    if cuda:
        sed_model.cuda()

    '''
    at_checkpoint_path = '/mnt/cephfs_hl/speechsv/qiuqiang.kong/workspaces/audioset_tagging_cnn_transfer/checkpoints/main_segment/sample_rate=32000,window_size=1024,hop_size=500,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13_GMP_fromwaveform_no_scale_c_specaug_64x64/loss_type=bce/balanced=balanced/augmentation=none/batch_size=32/accumulation_steps=1/980000_iterations.pth'
    '''
    at_checkpoint_path = '/mnt/cephfs_hl/speechsv/qiuqiang.kong/workspaces/audioset_tagging_cnn_transfer/checkpoints/main4/sample_rate=32000,window_size=1024,hop_size=500,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13_GMP_fromwaveform_no_scale_c_specaug_64x64/loss_type=bce/balanced=balanced/augmentation=none/batch_size=32/accumulation_steps=1/3200000_iterations.pth'

    checkpoint = torch.load(at_checkpoint_path)
    sed_model.load_state_dict(checkpoint['model'])
    
    # SS Model
    Model = eval(model_type)
    model = Model(classes_num)
    
    params_num = count_parameters(model)
    logging.info('Parameters number: {}'.format(params_num))

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if cuda:
        model.cuda()

    ss_checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'accumulation_steps={}'.format(accumulation_steps), 
        '{}_iterations.pth'.format(iteration))
        
    checkpoint = torch.load(ss_checkpoint_path)
    model.load_state_dict(checkpoint['model'])
 
    ###
    # _mixture = np.ones(config.sample_rate * 10)[None, :] * 1e-9
    (_mixture, _) = librosa.core.load('ss_wav_new/RrS0prSV8kc.wav', sr=None, mono=False)
    _mixture += 1e-10
    _mixture = _mixture[None, :]
    mixture = move_data_to_device(_mixture, device)

    with torch.no_grad():
        sed_model.eval()
        prediction = sed_model(mixture)

    (sorted_p, indices) = torch.sort(prediction, dim=-1, descending=True)

    sub_folder = 'ss_wav_new'
    sub_folder2 = 'ss_wav_new527'
    create_folder(sub_folder)
    create_folder(sub_folder2)
    librosa.output.write_wav(os.path.join(sub_folder, 'mixture.wav'), _mixture[0], sr=sample_rate)

    indices = indices.data.cpu().numpy()
    # indices = np.array([[0, 187, 191, 193, 198]])

    for i in range(5):
        indice = indices[0, i]
        print('{}: {}'.format(config.ix_to_lb[indice], sorted_p.data.cpu().numpy()[0, i]))

        classes_num = prediction.shape[-1]
        class_id = prediction
        
        with torch.no_grad():
            model.eval()
            batch_spec_output = model(mixture, class_id)
            length = mixture.shape[-1]
            output = model.module.wavin_to_wavout(mixture, class_id, length=length)
            output = output.data.cpu().numpy()[0]

        librosa.output.write_wav(os.path.join(sub_folder, 'sep_{}.wav'.format(i)), output, sr=sample_rate)

    for indice in range(classes_num):
        print('{} {}: {}'.format(indice, config.ix_to_lb[indice], prediction.data.cpu().numpy()[0, indice]))

        classes_num = prediction.shape[-1]
        class_id = torch.zeros(1, classes_num).to(device)
        class_id[0, indice] = 1.
        # class_id[0, indice] = prediction[0, indice]
        
        with torch.no_grad():
            model.eval()
            batch_spec_output = model(mixture, class_id)
            length = mixture.shape[-1]
            output = model.module.wavin_to_wavout(mixture, class_id, length=length)
            output = output.data.cpu().numpy()[0]

        librosa.output.write_wav(os.path.join(sub_folder2, 'sep_{}_{}.wav'.format(indice, config.ix_to_lb[indice])), output, sr=sample_rate)

    print('Write to {}'.format(sub_folder))
"""


def inference_new(args):

    dataset_dir = args.dataset_dir
    workspace = args.workspace
    workspace2 = args.workspace2
    data_type = args.data_type
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    iteration = args.iteration
    accumulation_steps = args.accumulation_steps
    cuda = args.cuda and torch.cuda.is_available()
    filename = args.filename

    classes_num = config.classes_num
    sample_rate = config.sample_rate
    total_samples = sample_rate * config.clip_seconds
    total_frames = total_samples // hop_size
    num_workers = 0

    # Paths
    hdf5s_dir = os.path.join(dataset_dir, 'features', 'waveform_hdf5')
 
    black_list_csv = os.path.join(workspace, 'black_list', 'dcase2017task4.csv')
    
    class_labels_indices_path = os.path.join(dataset_dir, 'metadata', 
        'class_labels_indices.csv')

    test_targets_hdf5_path = os.path.join(dataset_dir, 'coarse_targets_hdf5', 
        'eval_target.h5')

    # test_targets_hdf5_path = os.path.join(dataset_dir, 'coarse_targets_hdf5', 
    #     'balanced_train_target.h5')

    if cuda:
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'
    
    # Load labels
    (ids, labels) = load_class_label_indices(class_labels_indices_path)
    classes_num = len(labels)
    
    sed_model = Cnn13_GMPGAP_fromwaveform_no_scale_c_specaug_100x64_Fc_DecisionLevelMax(
        window_size=window_size, hop_size=hop_size, window=window, 
        center=center, pad_mode=pad_mode, sample_rate=sample_rate, 
        mel_bins=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, 
        top_db=top_db, freeze_parameters=True, scalar=None, 
        classes_num=classes_num, device=device)

    at_checkpoint_path = '/mnt/cephfs_new_wj/speechsv/qiuqiang.kong/workspaces/audioset_tagging_cnn_transfer/checkpoints/main7/sample_rate=32000,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000/data_type=full_train/Cnn13_GMPGAP_fromwaveform_no_scale_c_specaug_100x64_Fc_DecisionLevelMax/loss_type=clip_bce/balanced=balanced/augmentation=none/batch_size=32/accumulation_steps=1/200000_iterations.pth'


    # sed_model = nn.DataParallel(sed_model)    

    if cuda:
        sed_model.cuda()
        
    checkpoint = torch.load(at_checkpoint_path)
    sed_model.load_state_dict(checkpoint['model'])

    sed_model = nn.DataParallel(sed_model)

    # SS Model
    Model = eval(model_type)
    model = Model(classes_num)
    
    params_num = count_parameters(model)
    logging.info('Parameters number: {}'.format(params_num))

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if cuda:
        model.cuda()

    ss_checkpoint_path = os.path.join(workspace2, 'checkpoints', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'accumulation_steps={}'.format(accumulation_steps), '{}_iterations.pth'.format(iteration))
         
    checkpoint = torch.load(ss_checkpoint_path)
    model.load_state_dict(checkpoint['model'])
 
    ###
    # _mixture = np.ones(config.sample_rate * 10)[None, :] * 1e-9
    (_mixture, _) = librosa.core.load('ss_wav_new/s4pxJ-1QLQg.wav', sr=None, mono=False)
    _mixture += 1e-10



    # (sorted_p, indices) = torch.sort(prediction, dim=-1, descending=True)

    # sub_folder = 'ss_wav_new'
    sub_folder2 = 'ss_wav_new527'
    create_folder(sub_folder2)
    librosa.output.write_wav(os.path.join(sub_folder2, 'mixture.wav'), _mixture, sr=sample_rate)

    sub_folder_tops = 'ss_wav_new_tops'
    create_folder(sub_folder_tops)
    librosa.output.write_wav(os.path.join(sub_folder_tops, 'mixture.wav'), _mixture, sr=sample_rate)

    N = len(_mixture) // 320000
    _mixture = _mixture.reshape((N, 320000))
    mixture = move_data_to_device(_mixture, device)

    with torch.no_grad():
        sed_model.eval()
        output_dict = sed_model(mixture)

    clipwise_output = output_dict['clipwise_output'].data.cpu().numpy()[0]
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    tops_num = 10
    sorted_indexes = sorted_indexes[0 : tops_num]
    for i in range(tops_num):
        print('{}: {:.3f}'.format(config.ix_to_lb[sorted_indexes[i]], clipwise_output[sorted_indexes[i]]))
 
    print('--------------')

    hard_cond = torch.zeros(N, classes_num).to(device)
    soft_cond = torch.zeros(N, classes_num).to(device)

    with torch.no_grad():
        model.eval()
        batch_spec_output = model(mixture, hard_cond, soft_cond)
        length = mixture.shape[-1]
        output = model.module.wavin_to_wavout(mixture, hard_cond, soft_cond, length=length)
        output = output.data.cpu().numpy()
    output = np.concatenate(output, axis=0)
    librosa.output.write_wav(os.path.join(sub_folder2, 'sep_zero.wav'.format(-1, 'zero')), output, sr=sample_rate)

    for indice in range(classes_num):
        

        # classes_num = prediction.shape[-1]
        # class_id = torch.zeros(1, classes_num).to(device)
        # class_id[0, indice] = 1.
        # class_id[0, indice] = prediction[0, indice]

        if indice in sorted_indexes:
            print('{} {}'.format(indice, config.ix_to_lb[indice]))
            hard_cond = torch.zeros(N, classes_num).to(device)
            soft_cond = torch.zeros(N, classes_num).to(device)
 
            for i in range(N):
                # hard_cond[i, indice] = prediction[i, indice]
                hard_cond[i, indice] = 1
                # soft_cond[i, indice] = float(clipwise_output[indice])
                soft_cond[i, indice] = 1

            with torch.no_grad():
                model.eval()
                
                batch_spec_output = model(mixture, hard_cond, soft_cond)
                length = mixture.shape[-1]
                
                output = model.module.wavin_to_wavout(mixture, hard_cond, soft_cond, length=length)
                output = output.data.cpu().numpy()

            output = np.concatenate(output, axis=0)
 
            librosa.output.write_wav(os.path.join(sub_folder2, 'sep_{}_{}.wav'.format(indice, config.ix_to_lb[indice])), output, sr=sample_rate)

        if indice in sorted_indexes:
            librosa.output.write_wav(os.path.join(sub_folder_tops, 'sep_{}_{}.wav'.format(indice, config.ix_to_lb[indice])), output, sr=sample_rate)

    print('Write to {}'.format(sub_folder2))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')
 
    parser_train = subparsers.add_parser('train')  
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--at_checkpoint_path', type=str, required=True)
    parser_train.add_argument('--data_type', type=str, required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--condition_type', type=str, choices=['soft', 'soft_hard'], required=True)
    parser_train.add_argument('--wiener_filter', action='store_true', default=False)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--early_stop', type=int, required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--resume_iteration', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--dataset_dir', type=str, required=True)    
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--data_type', type=str, required=True)
    parser_inference.add_argument('--window_size', type=int, required=True)
    parser_inference.add_argument('--hop_size', type=int, required=True)
    parser_inference.add_argument('--mel_bins', type=int, required=True)
    parser_inference.add_argument('--fmin', type=int, required=True)
    parser_inference.add_argument('--fmax', type=int, required=True)
    parser_inference.add_argument('--model_type', type=str, required=True)
    parser_inference.add_argument('--loss_type', type=str, required=True)
    parser_inference.add_argument('--balanced', type=str, required=True)
    parser_inference.add_argument('--augmentation', type=str, required=True)
    parser_inference.add_argument('--batch_size', type=int, required=True)
    parser_inference.add_argument('--accumulation_steps', type=int, required=True)
    parser_inference.add_argument('--iteration', type=int, required=True)
    parser_inference.add_argument('--cuda', action='store_true', default=False)
    
    parser_evaluate_all = subparsers.add_parser('evaluate_all')
    parser_evaluate_all.add_argument('--dataset_dir', type=str, required=True)    
    parser_evaluate_all.add_argument('--workspace', type=str, required=True)
    parser_evaluate_all.add_argument('--workspace2', type=str, required=True)
    parser_evaluate_all.add_argument('--data_type', type=str, required=True)
    parser_evaluate_all.add_argument('--window_size', type=int, required=True)
    parser_evaluate_all.add_argument('--hop_size', type=int, required=True)
    parser_evaluate_all.add_argument('--mel_bins', type=int, required=True)
    parser_evaluate_all.add_argument('--fmin', type=int, required=True)
    parser_evaluate_all.add_argument('--fmax', type=int, required=True)
    parser_evaluate_all.add_argument('--model_type', type=str, required=True)
    parser_evaluate_all.add_argument('--loss_type', type=str, required=True)
    parser_evaluate_all.add_argument('--balanced', type=str, required=True)
    parser_evaluate_all.add_argument('--augmentation', type=str, required=True)
    parser_evaluate_all.add_argument('--batch_size', type=int, required=True)
    parser_evaluate_all.add_argument('--accumulation_steps', type=int, required=True)
    parser_evaluate_all.add_argument('--iteration', type=int, required=True)
    parser_evaluate_all.add_argument('--cuda', action='store_true', default=False)

    parser_combine_sdrs = subparsers.add_parser('combine_sdrs')
    parser_combine_sdrs.add_argument('--dataset_dir', type=str, required=True)    
    parser_combine_sdrs.add_argument('--workspace2', type=str, required=True)
    parser_combine_sdrs.add_argument('--data_type', type=str, required=True)
    parser_combine_sdrs.add_argument('--window_size', type=int, required=True)
    parser_combine_sdrs.add_argument('--hop_size', type=int, required=True)
    parser_combine_sdrs.add_argument('--mel_bins', type=int, required=True)
    parser_combine_sdrs.add_argument('--fmin', type=int, required=True)
    parser_combine_sdrs.add_argument('--fmax', type=int, required=True)
    parser_combine_sdrs.add_argument('--model_type', type=str, required=True)
    parser_combine_sdrs.add_argument('--loss_type', type=str, required=True)
    parser_combine_sdrs.add_argument('--balanced', type=str, required=True)
    parser_combine_sdrs.add_argument('--augmentation', type=str, required=True)
    parser_combine_sdrs.add_argument('--batch_size', type=int, required=True)
    parser_combine_sdrs.add_argument('--accumulation_steps', type=int, required=True)
    parser_combine_sdrs.add_argument('--bgn_iteration', type=int, required=True)
    parser_combine_sdrs.add_argument('--fin_iteration', type=int, required=True)
    parser_combine_sdrs.add_argument('--cuda', action='store_true', default=False)

    parser_inference_new = subparsers.add_parser('inference_new')
    parser_inference_new.add_argument('--dataset_dir', type=str, required=True)    
    parser_inference_new.add_argument('--workspace', type=str, required=True)
    parser_inference_new.add_argument('--workspace2', type=str, required=True)
    parser_inference_new.add_argument('--data_type', type=str, required=True)
    parser_inference_new.add_argument('--window_size', type=int, required=True)
    parser_inference_new.add_argument('--hop_size', type=int, required=True)
    parser_inference_new.add_argument('--mel_bins', type=int, required=True)
    parser_inference_new.add_argument('--fmin', type=int, required=True)
    parser_inference_new.add_argument('--fmax', type=int, required=True)
    parser_inference_new.add_argument('--model_type', type=str, required=True)
    parser_inference_new.add_argument('--loss_type', type=str, required=True)
    parser_inference_new.add_argument('--balanced', type=str, required=True)
    parser_inference_new.add_argument('--augmentation', type=str, required=True)
    parser_inference_new.add_argument('--batch_size', type=int, required=True)
    parser_inference_new.add_argument('--accumulation_steps', type=int, required=True)
    parser_inference_new.add_argument('--iteration', type=int, required=True)
    parser_inference_new.add_argument('--cuda', action='store_true', default=False)
 
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference':
        inference(args)

    elif args.mode == 'evaluate_all':
        evaluate_all(args)

    elif args.mode == 'combine_sdrs':
        combine_sdrs(args)

    elif args.mode == 'inference_new':
        inference_new(args)

    else:
        raise Exception('Error argument!')