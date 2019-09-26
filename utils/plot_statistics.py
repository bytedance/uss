import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../pytorch'))
import numpy as np
import argparse
import h5py
import _pickle as cPickle
import matplotlib.pyplot as plt
import torch

from sed_models import Cnn13_DecisionLevelMax
from pytorch_utils import SedMix
from data_generator import SsAudioSetDataset, SsBalancedSampler, collect_fn 
from utilities import (create_folder, get_filename)
import config


def plot(args):
    
    # Arguments & parameters
    workspace = args.workspace
    select = args.select
        
    save_out_path = 'results_map/{}.pdf'.format(select)
    create_folder(os.path.dirname(save_out_path))
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    lines = []
        
    def _load_metrics(filename, data_type, model_type, condition_type, 
        wiener_filter, loss_type, batch_size):
        statistics_path = os.path.join(workspace, 'statistics', filename, 
            'data_type={}'.format(data_type), model_type, 
            'condition_type={}'.format(condition_type), 'wiener_filter={}'.format(wiener_filter), 
            'loss_type={}'.format(loss_type), 'batch_size={}'.format(batch_size), 
            'statistics.pkl')

        statistics_dict = cPickle.load(open(statistics_path, 'rb'))
        bal_sdr = np.array([stat['sdr'] for stat in statistics_dict['bal']])
        test_sdr = np.array([stat['sdr'] for stat in statistics_dict['test']])
        return bal_sdr, test_sdr
        
    bal_alpha = 0.3
    test_alpha = 1.0
    lines = []
          
    if select == '1':
        (bal_sdr, test_sdr) = _load_metrics('ss_main', 'balanced_train', 'UNet', 'soft', False, 'mae', 12)
        line, = ax.plot(bal_sdr, label='UNet_soft_no_wiener', color='r', alpha=bal_alpha)
        line, = ax.plot(test_sdr, label='UNet_soft_no_wiener', color='r', alpha=test_alpha)
        lines.append(line)

        (bal_sdr, test_sdr) = _load_metrics('ss_main', 'balanced_train', 'UNet', 'soft', True, 'mae', 12)
        line, = ax.plot(bal_sdr, label='UNet_soft_wiener', color='b', alpha=bal_alpha)
        line, = ax.plot(test_sdr, label='UNet_soft_wiener', color='b', alpha=test_alpha)
        lines.append(line)

        (bal_sdr, test_sdr) = _load_metrics('ss_main', 'balanced_train', 'UNet', 'soft_hard', False, 'mae', 12)
        line, = ax.plot(bal_sdr, label='UNet_soft_hard_no_wiener', color='g', alpha=bal_alpha)
        line, = ax.plot(test_sdr, label='UNet_soft_hard_no_wiener', color='g', alpha=test_alpha)
        lines.append(line)

        (bal_sdr, test_sdr) = _load_metrics('ss_main', 'balanced_train', 'UNet', 'soft_hard', True, 'mae', 12)
        line, = ax.plot(bal_sdr, label='UNet_soft_hard_wiener', color='k', alpha=bal_alpha)
        line, = ax.plot(test_sdr, label='UNet_soft_hard_wiener', color='k', alpha=test_alpha)
        lines.append(line)

    iterations = np.arange(0, 1000000, 2000)
    max_plot_iteration = 1000000
    ax.set_ylim(-5, 10)
    ax.set_xlim(0, len(iterations))
    ax.xaxis.set_ticks(np.arange(0, len(iterations), 25))
    ax.xaxis.set_ticklabels(np.arange(0, max_plot_iteration, 50000))
    # ax.yaxis.set_ticks(np.arange(0, 1.01, 0.05))
    # ax.yaxis.set_ticklabels(np.around(np.arange(0, 1.01, 0.05), decimals=2))        
    ax.grid(color='b', linestyle='solid', linewidth=0.3)
    plt.legend(handles=lines, loc=2)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(handles=lines, bbox_to_anchor=(1.0, 1.0))

    plt.savefig(save_out_path)
    print('Save figure to {}'.format(save_out_path))


def plot_waveform_sed(args):

    # Arugments & parameters
    workspace = args.workspace
    at_checkpoint_path = args.at_checkpoint_path
    batch_size = 12
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    filename = 'ss_main'

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

    # Dataset will be used by DataLoader later. Provide an index and return 
    # waveform and target of audio
    bal_dataset = SsAudioSetDataset(
        target_hdf5_path=eval_train_targets_hdf5_path, 
        waveform_hdf5s_dir=waveform_hdf5s_dir, 
        audio_length=audio_length, 
        classes_num=classes_num)

    # Sampler
    bal_sampler = SsBalancedSampler(
        target_hdf5_path=eval_train_targets_hdf5_path, 
        batch_size=batch_size)

    # Data loader
    bal_loader = torch.utils.data.DataLoader(dataset=bal_dataset, 
        batch_sampler=bal_sampler, collate_fn=collect_fn, 
        num_workers=num_workers, pin_memory=True)
    
    # Load pretrained SED model. Do not change these parameters as they are 
    # part of the pretrained SED model.
    sed_model = Cnn13_DecisionLevelMax(sample_rate=32000, window_size=1024, 
        hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)

    print('Loading checkpoint {}'.format(at_checkpoint_path))
    checkpoint = torch.load(at_checkpoint_path)
    sed_model.load_state_dict(checkpoint['model'])
    sed_model = torch.nn.DataParallel(sed_model)

    if 'cuda' in str(device):
        sed_model.to(device)

    sed_mix = SedMix(sed_model, neighbour_segs=neighbour_segs, sample_rate=sample_rate)

    for iteration, batch_10s_dict in enumerate(bal_loader):
        batch_data_dict = sed_mix.get_mix_data(batch_10s_dict, sed_model, with_identity_zero=False)
        print(np.array(config.labels)[batch_data_dict['class_id']])

        for j in range(batch_size):
            print(np.max(batch_data_dict['segmentwise_output'][j, :, batch_data_dict['class_id'][j]]))

        n1 = 3
        n2 = 4
        id1 = batch_10s_dict['class_id'][n1]
        id2 = batch_10s_dict['class_id'][n2]
        fig, axs = plt.subplots(4,1, sharex=False)
        axs[0].plot(batch_10s_dict['waveform'][n1])
        axs[0].set_title(config.labels[id1])
        axs[0].set_ylim(-1, 1)
        axs[0].set_ylabel('Amplitude')
        axs[0].set_xlim(0, sample_rate * 10)
        axs[0].xaxis.set_ticks(np.arange(0, sample_rate*10, sample_rate*10 - 1))
        axs[0].xaxis.set_ticklabels(['0', '10 s'])
        
        axs[1].plot(batch_data_dict['segmentwise_output'][n1, :, id1], c='r')
        axs[1].set_ylim(0, 1)
        axs[1].set_xlim(0, 30)
        axs[1].set_ylabel('SED prob')
        axs[1].xaxis.set_ticks(np.arange(0, 31, 30))
        axs[1].xaxis.set_ticklabels(['0', '10 s'])

        axs[2].plot(batch_10s_dict['waveform'][n2])
        axs[2].set_title(config.labels[id2])
        axs[2].set_ylim(-1, 1)
        axs[2].set_xlim(0, sample_rate * 10)
        axs[2].xaxis.set_ticks(np.arange(0, sample_rate*10, sample_rate*10 - 1))
        axs[2].xaxis.set_ticklabels(['0', '10 s'])

        axs[2].set_ylabel('Amplitude')
        axs[3].plot(batch_data_dict['segmentwise_output'][n2, :, id2], c='r')
        axs[3].set_ylabel('SED prob')
        axs[3].set_ylim(0, 1)
        axs[3].set_xlim(0, 30)
        axs[3].xaxis.set_ticks(np.arange(0, 31, 30))
        axs[3].xaxis.set_ticklabels(['0', '10 s'])
        plt.tight_layout(1, 0, 0)
        plt.savefig('_debug/waveform.pdf')

        if iteration == 1:
            import crash
            asdf


def prepare_plot(sorted_lbs, set_lim=False):

    N = len(sorted_lbs)
    
    truncated_sorted_lbs = []
    for lb in sorted_lbs:
        lb = lb[0 : 20]
        words = lb.split(' ')
        if len(words[-1]) < 3:
            lb = ' '.join(words[0:-1])
        truncated_sorted_lbs.append(lb)

    if set_lim:
        sharey = False
    else:
        sharey = True

    f,(ax,ax2) = plt.subplots(1,2,sharey=sharey, facecolor='w', figsize=(20, 6))

    ax.xaxis.set_ticks(np.arange(len(sorted_lbs)))
    ax.xaxis.set_ticklabels(truncated_sorted_lbs, rotation=90, fontsize=14)
    ax.xaxis.tick_bottom()
    ax.set_ylabel("Number of audio clips", fontsize=14)
    
    ax2.xaxis.set_ticks(np.arange(len(sorted_lbs)))
    ax2.xaxis.set_ticklabels(truncated_sorted_lbs, rotation=90, fontsize=14)
    ax2.xaxis.tick_bottom()

    # ax.set_yscale('log')
    # ax2.set_yscale('log')
    
    K = 30
    ax.set_xlim(0, K)
    ax2.set_xlim(N - K, N)

    d = .015 # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1-d,1+d), (-d,+d), **kwargs)
    ax.plot((1-d,1+d),(1-d,1+d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    ax2.plot((-d,+d), (-d,+d), **kwargs)
    
    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ax2.tick_params(left='off', which='both')
    
    # ax2.yaxis.set_visible(False)
    ax.set_ylim(-10, 30)
    ax.set_ylabel('SDR')
    ax.yaxis.grid(color='grey', linestyle='--', alpha=0.5)
    ax2.yaxis.grid(color='grey', linestyle='--', alpha=0.5)

    # if False:
    #     ax1b = ax.twinx()
    #     ax2b = ax2.twinx()
    #     ax1b.set_ylim(-20, 20)
    #     ax1b.set_ylabel('SDR')
  
    #     ax1b.spines['right'].set_visible(False)
    #     ax2b.spines['left'].set_visible(False)
    #     ax1b.tick_params(labelright='off')
    #     ax1b.tick_params(right='off')
        
    #     ax1b.yaxis.grid(color='grey', linestyle='--', alpha=0.5)
    #     ax2b.yaxis.grid(color='grey', linestyle='--', alpha=0.5)
        
    ax2.tick_params(left='off', which='both')
    plt.tight_layout()
    
    # return ax, ax1b, ax2, ax2b
    return ax, ax2, None, None


def _plot(x, ax, ax2):
    ax.boxplot(x, manage_ticks=False)
    ax2.boxplot(x, manage_ticks=False)
    line = None
    return line


def plot_long_fig(args):

    # Arguments & parameters
    workspace = args.workspace
    select = args.select

    classes_num = config.classes_num
    labels = config.labels

    # Paths
    save_out_path = 'results_map/long_fig.pdf'
    create_folder(os.path.dirname(save_out_path))

    def _load_results(filename, data_type, model_type, condition_type, wiener_filter, 
        loss_type, batch_size, iteration, eval_type):

        results_path = os.path.join(workspace, 'results', filename, 
            'data_type={}'.format(data_type), model_type, 
            'condition_type={}'.format(condition_type), 'wiener_filter={}'.format(wiener_filter), 
            'loss_type={}'.format(loss_type), 'batch_size={}'.format(batch_size), 
            '{}_iterations.pth'.format(iteration))

        results_dict = cPickle.load(open(results_path, 'rb'))
        results = results_dict[eval_type]

        return results

    results = _load_results('ss_main', 'balanced_train', 
        'UNet', 'soft_hard', True, 'mae', 12, 240000, 'test')

    sdr_dict = results['sdr']
    sdrs = []
    for k in range(classes_num):
        sdr_dict[k] = np.array(sdr_dict[k])
        sdr_dict[k][np.isnan(sdr_dict[k])] = 0
        sdr = np.mean(sdr_dict[k])
        sdrs.append(sdr)
    sdrs = np.array(sdrs)

    sorted_idxes = np.argsort(sdrs)[::-1]
    sorted_lbs = [labels[idx] for idx in sorted_idxes]

    sdrs_list = []
    for k in sorted_idxes:
        sdrs_list.append(sdr_dict[k])

    (ax, ax2, _, _) = prepare_plot(sorted_lbs)
    _plot(sdrs_list, ax, ax2)
    
    plt.savefig(save_out_path)
    print('Save fig to {}'.format(save_out_path))

    print('avg sdr: {:.3f}'.format(np.mean([np.mean(e) for e in sdrs_list])))

    import crash
    asdf
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_plot = subparsers.add_parser('plot')
    parser_plot.add_argument('--workspace', type=str, required=True)
    parser_plot.add_argument('--select', type=str, required=True)

    parser_plot_waveform_sed = subparsers.add_parser('plot_waveform_sed')
    parser_plot_waveform_sed.add_argument('--workspace', type=str, required=True)
    parser_plot_waveform_sed.add_argument('--at_checkpoint_path', type=str, required=True)
    
    parser_plot_long_fig = subparsers.add_parser('plot_long_fig')
    parser_plot_long_fig.add_argument('--workspace', type=str, required=True)
    parser_plot_long_fig.add_argument('--select', type=str, required=True)

    args = parser.parse_args()

    if args.mode == 'plot':
        plot(args)
         
    elif args.mode == 'plot_waveform_sed':
        plot_waveform_sed(args)

    elif args.mode == 'plot_long_fig':
        plot_long_fig(args)

    else:
        raise Exception('Error argument!')
