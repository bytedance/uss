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
import pickle

import torch
import torch.optim as optim
# from sed_models import Cnn13_DecisionLevelMax
from models import UNet
from utilities import (create_folder, get_filename, create_logging, StatisticsContainer)
import config
from losses import get_loss_func
from data_generator import SsAudioSetDataset, SsBalancedSampler, SsEvaluateSampler, collect_fn 
# from pytorch_utils import count_parameters, SedMix, debug_and_plot, move_data_to_device
from pytorch_utils import count_parameters, SedMix, move_data_to_device, id_to_one_hot
from panns_inference import SoundEventDetection, AudioTagging, labels
from evaluate import Evaluator, average_dict, calculate_sdr


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
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    mix_type = args.mix_type
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename

    num_workers = 8
    sample_rate = config.sample_rate
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    segment_frames = config.segment_frames
    loss_func = get_loss_func(loss_type)
    # max_iteration = 10
    max_iteration = int(np.ceil(classes_num * 50 / batch_size))
    # max_iteration = int(np.ceil(classes_num * 3 / batch_size))
# 
    if mix_type in ['4b']:
        condition_type = 'hard_condition'
    elif mix_type in ['3', '5', '5b']:
        condition_type = 'soft_condition'
    else:
        raise Exception('Incorrect mix_type!')

    # neighbour_segs = 2  # segments used for training has length of (neighbour_segs * 2 + 1) * 0.32 ~= 1.6 s
    # eval_max_iteration = 2  # Number of mini_batches for validation
    
    # Paths
    train_indexes_hdf5_path = os.path.join(workspace, 'hdf5s', 'indexes', 
        '{}.h5'.format(data_type))

    eval_bal_indexes_hdf5_path = os.path.join(workspace, 
        'hdf5s', 'indexes', 'balanced_train.h5')

    eval_test_indexes_hdf5_path = os.path.join(workspace, 'hdf5s', 'indexes', 
        'eval.h5')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'mix_type={}'.format(mix_type), 
        'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)
    
    statistics_path = os.path.join(workspace, 'statistics', filename, 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'mix_type={}'.format(mix_type), 
        'batch_size={}'.format(batch_size), 'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'mix_type={}'.format(mix_type), 
        'batch_size={}'.format(batch_size))

    create_logging(logs_dir, filemode='w')
    logging.info(args)
    
    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'

    dataset = SsAudioSetDataset(clip_samples=clip_samples, classes_num=classes_num)

    # Sampler
    train_sampler = SsBalancedSampler(
        indexes_hdf5_path=train_indexes_hdf5_path, 
        batch_size=batch_size)
        
    eval_bal_sampler = SsEvaluateSampler(
        indexes_hdf5_path=eval_bal_indexes_hdf5_path, 
        batch_size=batch_size, max_iteration=max_iteration)

    eval_test_sampler = SsEvaluateSampler(
        indexes_hdf5_path=eval_test_indexes_hdf5_path, 
        batch_size=batch_size, max_iteration=max_iteration)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=train_sampler, collate_fn=collect_fn, 
        num_workers=num_workers, pin_memory=True)

    eval_bal_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=eval_bal_sampler, collate_fn=collect_fn, 
        num_workers=num_workers, pin_memory=True)

    eval_test_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=eval_test_sampler, collate_fn=collect_fn, 
        num_workers=num_workers, pin_memory=True)
    
    # Load pretrained SED model. Do not change these parameters as they are 
    # part of the pretrained SED model.
    # sed_model = Cnn13_DecisionLevelMax(sample_rate=32000, window_size=1024, 
        # hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)

    

    # logging.info('Loading checkpoint {}'.format(at_checkpoint_path))
    # checkpoint = torch.load(at_checkpoint_path)
    # sed_model.load_state_dict(checkpoint['model'])
    # sed_model = torch.nn.DataParallel(sed_model)

    # if 'cuda' in str(device):
    #     sed_model.to(device)

    # Source separation model
    SsModel = eval(model_type)
    ss_model = SsModel(channels=1)
    
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
    
    sed_checkpoint_path = '/home/tiger/released_models/sed/Cnn14_DecisionLevelMax_mAP=0.385.pth'
    at_checkpoint_path = '/home/tiger/released_models/sed/Cnn14_mAP=0.431.pth'
    # sed_checkpoint_path = '/root/released_models/sed/Cnn14_DecisionLevelMax_mAP=0.385.pth'
    # at_checkpoint_path = '/root/released_models/sed/Cnn14_mAP=0.431.pth'

    sed_model = SoundEventDetection(device=device, checkpoint_path=sed_checkpoint_path)
    at_model = AudioTagging(device=device, checkpoint_path=at_checkpoint_path)
    sed_mix = SedMix(sed_model, at_model, segment_frames=segment_frames, sample_rate=sample_rate)

    evaluator = Evaluator(sed_mix=sed_mix, ss_model=ss_model, condition_type=condition_type)

    '''
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
    '''
    train_bgn_time = time.time()
    t1 = time.time()

    for batch_10s_dict in train_loader:
        
        # Evaluate  
        if (iteration % 20000 == 0 and iteration > resume_iteration) or (iteration == 0):
            train_fin_time = time.time()

            bal_statistics = evaluator.evaluate(eval_bal_loader) 
            test_statistics = evaluator.evaluate(eval_test_loader)

            logging.info('mixture si-sdr: {:.3f}, clean si-sdr: {:.3f}, silence sdr: {:.3f}'.format(
                average_dict(bal_statistics['mixture_sdr']), 
                average_dict(bal_statistics['clean_sdr']), 
                average_dict(bal_statistics['silence_sdr'])))

            logging.info('mixture si-sdr: {:.3f}, clean si-sdr: {:.3f}, silence sdr: {:.3f}'.format(
                average_dict(test_statistics['mixture_sdr']), 
                average_dict(test_statistics['clean_sdr']), 
                average_dict(test_statistics['silence_sdr'])))

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
        if iteration % 50000 == 0:
            checkpoint = {
                'iteration': iteration, 
                'model': ss_model.module.state_dict(), 
                'sampler': train_sampler.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
        
        # Get mixture and target data
        if mix_type == '1':
            batch_data_dict = sed_mix.get_mix_data(batch_10s_dict)
        elif mix_type == '2':
            batch_data_dict = sed_mix.get_mix_data2(batch_10s_dict)
        elif mix_type == '3':
            batch_data_dict = sed_mix.get_mix_data3(batch_10s_dict)
        elif mix_type == '4':
            batch_data_dict = sed_mix.get_mix_data4(batch_10s_dict)
        elif mix_type == '4b':
            batch_data_dict = sed_mix.get_mix_data4b(batch_10s_dict)
        elif mix_type == '5':
            batch_data_dict = sed_mix.get_mix_data5(batch_10s_dict)
        elif mix_type == '5b':
            batch_data_dict = sed_mix.get_mix_data5b(batch_10s_dict)

        if batch_data_dict:
            if False:
                import crash
                asdf
                K = 3
                config.ix_to_lb[batch_data_dict['class_id'][K]]
                batch_data_dict['hard_condition'][K]
                librosa.output.write_wav('_zz.wav', batch_data_dict['mixture'][K], sr=32000)
                librosa.output.write_wav('_zz2.wav', batch_data_dict['source'][K], sr=32000)
            
            if False:
                debug_and_plot(ss_model, batch_10s_dict, batch_data_dict)
                
            # Move data to device
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

            # Forward
            ss_model.train()
            if mix_type in ['1', '2', '4', '4b']:
                batch_output_dict = ss_model(batch_data_dict['mixture'], batch_data_dict['hard_condition'])
            elif mix_type in ['3', '5', '5b']:
                batch_output_dict = ss_model(batch_data_dict['mixture'], batch_data_dict['soft_condition'])

            loss = loss_func(batch_output_dict['wav'], batch_data_dict['source'])
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


def print_stat(args):
    statistics_path = '/home/tiger/workspaces/audioset_source_separation/statistics/ss_main/data_type=balanced_train/UNet/loss_type=mae/balanced=balanced/augmentation=none/batch_size=12/statistics.pkl'

    stat_dict = pickle.load(open(statistics_path, 'rb'))
    average_dict(stat_dict['test'][-1]['sdr'])

    tmp_dict = stat_dict['test'][-1]['sdr']
    values = []

    for key in range(config.classes_num):
        values.append(np.mean(tmp_dict[key]))

    values = np.array(values)
    indexes = np.argsort(values)[::-1]

    plt.figure(figsize=(12, 2))
    plt.plot(np.zeros(config.classes_num))
    plt.plot(values[indexes])
    plt.xticks(np.arange(config.classes_num), np.array(config.labels)[indexes], rotation='270', fontsize=2)
    # plt.xticks(np.arange(config.classes_num), np.arange(config.classes_num), rotation='270', fontsize=2)

    if True:
        tmp_dict = stat_dict['test'][5]['sdr']
        values = []
        for key in range(config.classes_num):
            values.append(np.mean(tmp_dict[key]))
        values = np.array(values)
        indexes = np.argsort(values)[::-1]
        plt.plot(values[indexes])

    plt.tight_layout()
    plt.savefig('_zz.pdf')

    import crash
    asdf

    

def validate(args):
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
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    iteration = args.iteration
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename

    num_workers = 8
    sample_rate = config.sample_rate
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    segment_frames = config.segment_frames
    loss_func = get_loss_func(loss_type)
    max_iteration = int(np.ceil(classes_num * 50 / batch_size))

    # neighbour_segs = 2  # segments used for training has length of (neighbour_segs * 2 + 1) * 0.32 ~= 1.6 s
    # eval_max_iteration = 2  # Number of mini_batches for validation
    
    # Paths
    eval_bal_indexes_hdf5_path = os.path.join(workspace, 
        'hdf5s', 'indexes', 'balanced_train.h5')

    eval_test_indexes_hdf5_path = os.path.join(workspace, 'hdf5s', 'indexes', 
        'eval.h5')

    '''
    checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        '{}_iterations.pth'.format(iteration))
    '''
    checkpoint_path = '/home/tiger/workspaces/audioset_source_separation/checkpoints/ss_main/data_type=balanced_train/UNet/loss_type=mae/balanced=balanced/augmentation=none/batch_size=12/900000_iterations.pth'
    # checkpoint_path = '/home/tiger/workspaces/audioset_source_separation/checkpoints/ss_main/data_type=balanced_train/UNet/loss_type=mae/balanced=balanced/augmentation=none/mix_type=2/batch_size=12/160000_iterations.pth'


    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'

    dataset = SsAudioSetDataset(clip_samples=clip_samples, classes_num=classes_num)

    # Sampler
    eval_bal_sampler = SsEvaluateSampler(
        indexes_hdf5_path=eval_bal_indexes_hdf5_path, 
        batch_size=batch_size, max_iteration=max_iteration)

    eval_test_sampler = SsEvaluateSampler(
        indexes_hdf5_path=eval_test_indexes_hdf5_path, 
        batch_size=batch_size, max_iteration=max_iteration)

    # Data loader
    eval_bal_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=eval_bal_sampler, collate_fn=collect_fn, 
        num_workers=num_workers, pin_memory=True)

    eval_test_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=eval_test_sampler, collate_fn=collect_fn, 
        num_workers=num_workers, pin_memory=True)
    
    # Source separation model
    SsModel = eval(model_type)
    ss_model = SsModel(channels=1)
    
    # Resume training
    checkpoint = torch.load(checkpoint_path)
    ss_model.load_state_dict(checkpoint['model'])
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    ss_model = torch.nn.DataParallel(ss_model)

    if 'cuda' in str(device):
        ss_model.to(device)
    
    sed_checkpoint_path = '/home/tiger/released_models/sed/Cnn14_DecisionLevelMax_mAP=0.385.pth'
    at_checkpoint_path = '/home/tiger/released_models/sed/Cnn14_mAP=0.431.pth'
    sed_model = SoundEventDetection(device=device, checkpoint_path=sed_checkpoint_path)
    at_model = AudioTagging(device=device, checkpoint_path=at_checkpoint_path)
    sed_mix = SedMix(sed_model, at_model, segment_frames=segment_frames, sample_rate=sample_rate)

    evaluator = Evaluator(sed_mix=sed_mix, ss_model=ss_model)

    for iteration, batch_10s_dict in enumerate(eval_test_loader):
        if iteration % 10 == 0:
            print(iteration)

        audios_num = len(batch_10s_dict['audio_name'])

        # Get mixture and target data
        batch_data_dict = sed_mix.get_mix_data(batch_10s_dict)

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

        # Separate
        with torch.no_grad():
            ss_model.eval()

            # batch_data_dict['hard_condition'][:, :] = 0
            # batch_data_dict['hard_condition'][:, 0] = 1

            batch_output_dict = ss_model(
                batch_data_dict['mixture'], 
                batch_data_dict['hard_condition'])

            batch_sep_wavs = batch_output_dict['wav'].data.cpu().numpy()

        
        sdr_dict = {k: [] for k in range(classes_num)}
        for n in range(0, audios_num):
            sdr = calculate_sdr(batch_data_dict['source'].data.cpu().numpy()[n, :, 0], batch_sep_wavs[n, :, 0], scaling=True)
            # norm_sdr = sdr - calculate_sdr(
            #     batch_data_dict['source'].data.cpu().numpy()[n, :, 0], 
            #     batch_data_dict['mixture'].data.cpu().numpy()[n, :, 0], scaling=True)

            class_id = batch_data_dict['class_id'].data.cpu().numpy()[n]
            sdr_dict[class_id].append(sdr)

        import crash
        asdf
        K = 5
        calculate_sdr(batch_data_dict['source'].data.cpu().numpy()[K, :, 0], batch_sep_wavs[K, :, 0], scaling=True)
        calculate_sdr(batch_data_dict['source'].data.cpu().numpy()[K, :, 0], batch_data_dict['mixture'].data.cpu().numpy()[K, :, 0], scaling=True)
        librosa.output.write_wav('_zz.wav', batch_data_dict['source'].data.cpu().numpy()[K, :, 0], sr=32000)
        librosa.output.write_wav('_zz2.wav', batch_data_dict['mixture'].data.cpu().numpy()[K, :, 0], sr=32000)
        librosa.output.write_wav('_zz3.wav', batch_sep_wavs[K, :, 0], sr=32000)
        print(config.labels[batch_data_dict['class_id'][K]])


def print_audio_tagging_result(clipwise_output):
    """Visualization of audio tagging result.
    Args:
      clipwise_output: (classes_num,)
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{} {}: {:.3f}'.format(sorted_indexes[k], np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))


def inference_new(args):
    # Arugments & parameters
    workspace = args.workspace
    at_checkpoint_path = args.at_checkpoint_path
    data_type = args.data_type
    model_type = args.model_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    iteration = args.iteration
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename

    num_workers = 8
    sample_rate = config.sample_rate
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    segment_frames = config.segment_frames
    loss_func = get_loss_func(loss_type)
    max_iteration = int(np.ceil(classes_num * 50 / batch_size))

    # Paths
    # checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
    #     'data_type={}'.format(data_type), model_type, 
    #     'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
    #     'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
    #     '{}_iterations.pth'.format(iteration))
    # checkpoint_path = '/home/tiger/workspaces/audioset_source_separation/checkpoints/ss_main/data_type=balanced_train/UNet/loss_type=mae/balanced=balanced/augmentation=none/batch_size=12/1000000_iterations.pth'

    # checkpoint_path = '/root/workspaces/audioset_source_separation/checkpoints/ss_main/data_type=balanced_train/UNet/loss_type=mae/balanced=balanced/augmentation=none/mix_type=3/batch_size=12/520000_iterations.pth'
    # checkpoint_path = '/root/workspaces/audioset_source_separation/checkpoints/ss_main/data_type=balanced_train/UNet/loss_type=mae/balanced=balanced/augmentation=none/mix_type=4/batch_size=12/400000_iterations.pth'
    # checkpoint_path = '/root/workspaces/audioset_source_separation/checkpoints/ss_main/data_type=balanced_train/UNet/loss_type=mae/balanced=balanced/augmentation=none/mix_type=4b/batch_size=12/500000_iterations.pth'
    # checkpoint_path = '/home/tiger/workspaces/audioset_source_separation/checkpoints/ss_main/data_type=balanced_train/UNet/loss_type=mae/balanced=balanced/augmentation=none/mix_type=5/batch_size=12/100000_iterations.pth'
    # checkpoint_path = '/home/tiger/workspaces/audioset_source_separation/checkpoints/ss_main/data_type=full_train/UNet/loss_type=mae/balanced=balanced/augmentation=none/mix_type=5/batch_size=12/150000_iterations.pth'
    checkpoint_path = '/home/tiger/workspaces/audioset_source_separation/checkpoints/ss_main/data_type=balanced_train/UNet/loss_type=mae/balanced=balanced/augmentation=none/mix_type=5b/batch_size=12/200000_iterations.pth'

    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'

    # Source separation model
    SsModel = eval(model_type)
    ss_model = SsModel(channels=1)
    
    # Resume training
    checkpoint = torch.load(checkpoint_path)
    ss_model.load_state_dict(checkpoint['model'])
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    ss_model = torch.nn.DataParallel(ss_model)

    if 'cuda' in str(device):
        ss_model.to(device)
    
    sed_checkpoint_path = '/home/tiger/released_models/sed/Cnn14_DecisionLevelMax_mAP=0.385.pth'
    at_checkpoint_path = '/home/tiger/released_models/sed/Cnn14_mAP=0.431.pth'
    sed_model = SoundEventDetection(device=device, checkpoint_path=sed_checkpoint_path)
    at_model = AudioTagging(device=device, checkpoint_path=at_checkpoint_path)
    sed_mix = SedMix(sed_model, at_model, segment_frames=segment_frames, sample_rate=sample_rate)

    #
    # (audio, fs) = librosa.core.load('resources/vocals_accompaniment_10s.mp3', sr=32000, mono=True)
    # (audio, fs) = librosa.core.load('resources/beethoven_violin_sonata_20s.mp3', sr=32000, mono=True)
    (audio, fs) = librosa.core.load('resources/4.mp3', sr=32000, mono=True)
 
    (clipwise_output, embedding) = at_model.inference(audio[None, :])
    print_audio_tagging_result(clipwise_output[0])

    # id1 = 67
    id1 = 0
    # batch_data_dict = {'mixture': audio[None, :, None], 'hard_condition': id_to_one_hot(id1, classes_num)[None, :]}

    # hard_condition = id_to_one_hot(id1, classes_num)[None, :]
    # id1 = 0

    def _add(id1):
        hard_condition = id_to_one_hot(id1, classes_num)[None, :]

        batch_data_dict = {'mixture': audio[None, :, None], 'hard_condition': hard_condition}

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

        # Separate
        with torch.no_grad():
            ss_model.eval()

            batch_output_dict = ss_model(
                batch_data_dict['mixture'], 
                batch_data_dict['hard_condition'])

            batch_sep_wavs = batch_output_dict['wav'].data.cpu().numpy()

        K = 0
        librosa.output.write_wav('_zz.wav', batch_data_dict['mixture'].data.cpu().numpy()[K, :, 0], sr=32000)
        librosa.output.write_wav('_zz2.wav', batch_sep_wavs[K, :, 0], sr=32000)

    import crash
    asdf
    _add(67)
    # import crash
    # asdf
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')
 
    parser_train = subparsers.add_parser('train')  
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--at_checkpoint_path', type=str, required=True)
    parser_train.add_argument('--data_type', type=str, required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--balanced', type=str, default='balanced', choices=['balanced'])
    parser_train.add_argument('--augmentation', type=str, default='mixup', choices=['none'])
    parser_train.add_argument('--mix_type', type=str, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--early_stop', type=int, required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--resume_iteration', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    parser_print = subparsers.add_parser('print')

    parser_train = subparsers.add_parser('validate')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--at_checkpoint_path', type=str, required=True)
    parser_train.add_argument('--data_type', type=str, required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--balanced', type=str, default='balanced', choices=['balanced'])
    parser_train.add_argument('--augmentation', type=str, default='mixup', choices=['none'])
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--iteration', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    parser_inference_new = subparsers.add_parser('inference_new')
    parser_inference_new.add_argument('--workspace', type=str, required=True)
    parser_inference_new.add_argument('--at_checkpoint_path', type=str, required=True)
    parser_inference_new.add_argument('--data_type', type=str, required=True)
    parser_inference_new.add_argument('--model_type', type=str, required=True)
    parser_inference_new.add_argument('--loss_type', type=str, required=True)
    parser_inference_new.add_argument('--balanced', type=str, default='balanced', choices=['balanced'])
    parser_inference_new.add_argument('--augmentation', type=str, default='mixup', choices=['none'])
    parser_inference_new.add_argument('--batch_size', type=int, required=True)
    parser_inference_new.add_argument('--iteration', type=int, required=True)
    parser_inference_new.add_argument('--cuda', action='store_true', default=False)
     
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'print':
        print_stat(args)

    elif args.mode == 'validate':
        validate(args)

    elif args.mode == 'inference_new':
        inference_new(args)

    else:
        raise Exception('Error argument!')