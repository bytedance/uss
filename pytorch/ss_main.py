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


def validate(args):
    
    # Arugments & parameters
    workspace = args.workspace
    at_checkpoint_path = args.at_checkpoint_path
    data_type = args.data_type
    model_type = args.model_type
    condition_type = args.condition_type
    wiener_filter = args.wiener_filter
    loss_type = args.loss_type
    batch_size = args.batch_size
    iteration = args.iteration
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename

    num_workers = 8
    sample_rate = config.sample_rate
    audio_length = config.audio_length
    classes_num = config.classes_num
    neighbour_segs = 2  # segments used for training has length of (neighbour_segs * 2 + 1) * 0.32 ~= 1.6 s
    eval_max_iteration = 100  # Number of mini_batches for validation
    
    # Paths
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'waveforms')

    eval_train_targets_hdf5_path = os.path.join(workspace, 
        'hdf5s', 'targets', 'balanced_train.h5')

    eval_test_targets_hdf5_path = os.path.join(workspace, 'hdf5s', 'targets', 
        'eval.h5')

    ss_checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
        'data_type={}'.format(data_type), model_type, 
        'condition_type={}'.format(condition_type), 'wiener_filter={}'.format(wiener_filter), 
        'loss_type={}'.format(loss_type), 'batch_size={}'.format(batch_size), 
        '{}_iterations.pth'.format(iteration))

    results_path = os.path.join(workspace, 'results', filename, 
        'data_type={}'.format(data_type), model_type, 
        'condition_type={}'.format(condition_type), 'wiener_filter={}'.format(wiener_filter), 
        'loss_type={}'.format(loss_type), 'batch_size={}'.format(batch_size), 
        '{}_iterations.pth'.format(iteration))
    create_folder(os.path.dirname(results_path))
    
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
    bal_sampler = SsBalancedSampler(
        target_hdf5_path=eval_train_targets_hdf5_path, 
        batch_size=batch_size)

    test_sampler = SsBalancedSampler(
        target_hdf5_path=eval_test_targets_hdf5_path, 
        batch_size=batch_size)

    # Data loader
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

    # Load source separation model
    SsModel = eval(model_type)
    ss_model = SsModel(classes_num, condition_type, wiener_filter)
    logging.info('Loading source separation checkpoint {}'.format(at_checkpoint_path))
    checkpoint = torch.load(ss_checkpoint_path)
    ss_model.load_state_dict(checkpoint['model'])
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    ss_model = torch.nn.DataParallel(ss_model)

    if 'cuda' in str(device):
        ss_model.to(device)
    
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

    bal_results = bal_evaluator.calculate_result_dict()
    test_results = test_evaluator.calculate_result_dict()

    results_dict = {'bal': bal_results, 'test': test_results}
    cPickle.dump(results_dict, open(results_path, 'wb'))
    logging.info('Save results to {}'.format(results_path))


def inference_new(args):
    
    # Arugments & parameters
    workspace = args.workspace
    at_checkpoint_path = args.at_checkpoint_path
    data_type = args.data_type
    model_type = args.model_type
    condition_type = args.condition_type
    wiener_filter = args.wiener_filter
    loss_type = args.loss_type
    batch_size = args.batch_size
    iteration = args.iteration
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename
    new_audio_path = args.new_audio_path

    num_workers = 8
    sample_rate = config.sample_rate
    classes_num = config.classes_num
    neighbour_segs = 2  # segments used for training has length of (neighbour_segs * 2 + 1) * 0.32 ~= 1.6 s
    
    # Paths
    ss_checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
        'data_type={}'.format(data_type), model_type, 
        'condition_type={}'.format(condition_type), 'wiener_filter={}'.format(wiener_filter), 
        'loss_type={}'.format(loss_type), 'batch_size={}'.format(batch_size), 
        '{}_iterations.pth'.format(iteration))

    separated_wavs_dir = '_separated_wavs'
    create_folder(separated_wavs_dir)

    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'

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

    # Load source separation model
    SsModel = eval(model_type)
    ss_model = SsModel(classes_num, condition_type, wiener_filter)
    logging.info('Loading source separation checkpoint {}'.format(at_checkpoint_path))
    checkpoint = torch.load(ss_checkpoint_path)
    ss_model.load_state_dict(checkpoint['model'])
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    ss_model = torch.nn.DataParallel(ss_model)

    if 'cuda' in str(device):
        ss_model.to(device)
    
    if new_audio_path:
        (waveform, _) = librosa.core.load(new_audio_path, sr=sample_rate, mono=True)
    else:
        waveform = np.zeros(sample_rate * 10)
    
    out_mixture_path = os.path.join(separated_wavs_dir, 'mixture.wav')
    librosa.output.write_wav(out_mixture_path, waveform, sr=sample_rate)
    print('Write mixture to {}'.format(out_mixture_path))

    waveform = move_data_to_device(waveform, device)[None, :]
    
    with torch.no_grad():
        sed_model.eval()
        output_dict = sed_model(waveform)
        at_prediction = output_dict['clipwise_output'].data.cpu().numpy()[0]
        
    sorted_indexes = np.argsort(at_prediction)[::-1][0 : 20]
    np.array(config.labels)[sorted_indexes]
    at_prediction[sorted_indexes]

    for index in sorted_indexes:

        condition = np.zeros(classes_num)
        condition[index] = 1
        condition = move_data_to_device(condition, device)[None, :]
 
        with torch.no_grad():
            ss_model.eval()
            separated_wav = ss_model.module.wavin_to_wavout(
                waveform, condition, condition, 
                length=waveform.shape[-1]).data.cpu().numpy()[0]
            
            separated_wav[np.isnan(separated_wav)] = 0

            sep_audio_path = os.path.join(separated_wavs_dir, '{}_{}.wav'.format(index, config.labels[index]))
            librosa.output.write_wav(sep_audio_path, separated_wav, sr=sample_rate)
            print('{} {}, {:.3f}. Write separated wav to {}'.format(
                index, config.labels[index], at_prediction[index], sep_audio_path))


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

    parser_train = subparsers.add_parser('validate')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--at_checkpoint_path', type=str, required=True)
    parser_train.add_argument('--data_type', type=str, required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--condition_type', type=str, choices=['soft', 'soft_hard'], required=True)
    parser_train.add_argument('--wiener_filter', action='store_true', default=False)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--iteration', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    
    parser_inference_new = subparsers.add_parser('inference_new')
    parser_inference_new.add_argument('--workspace', type=str, required=True)
    parser_inference_new.add_argument('--at_checkpoint_path', type=str, required=True)
    parser_inference_new.add_argument('--data_type', type=str, required=True)
    parser_inference_new.add_argument('--model_type', type=str, required=True)
    parser_inference_new.add_argument('--condition_type', type=str, choices=['soft', 'soft_hard'], required=True)
    parser_inference_new.add_argument('--wiener_filter', action='store_true', default=False)
    parser_inference_new.add_argument('--loss_type', type=str, required=True)
    parser_inference_new.add_argument('--batch_size', type=int, required=True)
    parser_inference_new.add_argument('--iteration', type=int, required=True)
    parser_inference_new.add_argument('--cuda', action='store_true', default=False)
    parser_inference_new.add_argument('--new_audio_path', type=str, default='')

    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    if args.mode == 'validate':
        validate(args)

    elif args.mode == 'inference_new':
        inference_new(args)

    else:
        raise Exception('Error argument!')