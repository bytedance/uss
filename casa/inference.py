import os
import re
import librosa
import time
import pathlib
import pickle
import json
import soundfile
import argparse

from casa.utils import calculate_sdr
from casa.utils import create_logging, read_yaml, load_pretrained_model #, 

from casa.data.anchor_segment_detectors import AnchorSegmentDetector
from casa.data.anchor_segment_mixers import AnchorSegmentMixer
from casa.data.query_condition_extractors import QueryConditionExtractor
from casa.models.pl_modules import LitSeparation, get_model_class
from casa.models.resunet import *
from casa.config import IX_TO_LB, ID_TO_IX
from casa.parse_ontology import get_ontology_tree, get_subclass_indexes, Node
from casa.models.query_nets import initialize_query_net


ontology_path = 'metadata/ontology.json'
audioset632_id_to_lb = {}
with open(ontology_path) as f:
    data_list = json.load(f)
for e in data_list:
    audioset632_id_to_lb[e['id']] = e['name']


# def load_ss_model(at_model_name, ss_model_name, input_channels, output_channels, condition_type, condition_size, checkpoint_path, device):
def load_ss_model(configs, checkpoint_path, device):

    # at_model = load_pretrained_model(
    #     model_name=at_model_name,
    #     checkpoint_path=None,
    #     freeze=None,
    # )

    # query_net =

    # query_condition_extractor = QueryConditionExtractor(
    #     at_model=at_model,
    #     condition_type=condition_type,
    # )

    ss_model_type = configs["ss_model"]["model_type"]
    input_channels = configs["ss_model"]["input_channels"]
    output_channels = configs["ss_model"]["output_channels"]
    condition_size = configs['query_net']['outputs_num']

    query_net = initialize_query_net(
        configs=configs,
    )

    # Model = eval(ss_model_name)
    SsModel = get_model_class(model_type=ss_model_type)

    ss_model = SsModel(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
    )
    
    ss_model = SsModel(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
    )

    pl_model = LitSeparation.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False,
        ss_model=ss_model,
        anchor_segment_detector=None,
        anchor_segment_mixer=None,
        query_net=query_net,
        loss_function=None,
        optimizer_type=None,
        learning_rate=None,
        lr_lambda_func=None,
    ).to(device)

    return pl_model


def get_nodes_with_level_n(nodes, level):

    nodes_level_n = []

    for node in nodes:
        if node.level == level:
            nodes_level_n.append(node)

    return nodes_level_n


def get_children_indexes(node):
    
    nodes_level_n_children = Node.traverse(node=node)

    subclass_indexes = [ID_TO_IX[node.class_id] for node in nodes_level_n_children if node.class_id in ID_TO_IX]

    return subclass_indexes

'''
def get_segment_predictions(audio, segment_samples, query_net, device):

    audio_samples = audio.shape[-1]
    pointer = 0
    query_conditions = []

    while pointer < audio_samples:

        segment = librosa.util.fix_length(
            data=audio[pointer : pointer + segment_samples],
            size=segment_samples,
            axis=0,
        )
        
        segments = torch.Tensor(segment)[None, :].to(device)

        with torch.no_grad():
            query_net.eval()

            query_condition = query_net(
                segments=segments,
            ).squeeze().data.cpu().numpy()

        query_conditions.append(query_condition)
        pointer += segment_samples

    query_conditions = np.stack(query_conditions, axis=0)

    return query_conditions
'''

def calculate_segment_at_probs(audio, segment_samples, at_model, device):

    audio_samples = audio.shape[-1]
    pointer = 0
    # query_net_bottlenecks = []
    at_probs = []

    while pointer < audio_samples:

        segment = librosa.util.fix_length(
            data=audio[pointer : pointer + segment_samples],
            size=segment_samples,
            axis=0,
        )
        
        segments = torch.Tensor(segment)[None, :].to(device)

        with torch.no_grad():
            at_model.eval()
            at_prob = at_model(input=segments)['clipwise_output'].squeeze().data.cpu().numpy()

            # query_net.eval()
            # query_net_bottleneck = query_net.forward_base(
            #     source=segments,
            # ).squeeze().data.cpu().numpy()
            # (527,)

        at_probs.append(at_prob)
        # query_net_bottlenecks.append(query_net_bottleneck)

        pointer += segment_samples

    at_probs = np.stack(at_probs, axis=0)
    # query_net_bottlenecks = np.stack(query_net_bottlenecks, axis=0)

    # return at_probs, query_net_bottlenecks
    return at_probs

'''
def separate_by_query_conditions(audio, segment_samples, query_conditions, subclass_indexes, ss_model, device):

    audio_samples = audio.shape[-1]
    pointer = 0
    segment_index = 0
    classes_num = query_conditions.shape[-1]
    at_threshold = 0.2
    sep_audio = []

    while pointer < audio_samples:

        condition = np.zeros(classes_num)
        condition[subclass_indexes] = query_conditions[segment_index, subclass_indexes]

        max_prob = np.max(condition)

        if max_prob > at_threshold:

            segment = librosa.util.fix_length(
                data=audio[pointer : pointer + segment_samples],
                size=segment_samples,
                axis=0,
            )

            input_dict = {
                'mixture': torch.Tensor(segment[None, None, :]).to(device),
                'condition': torch.Tensor(condition[None, :]).to(device),
            }

            with torch.no_grad():
                ss_model.eval()
                output_dict = ss_model(input_dict=input_dict)
            
            sep_segment = output_dict['waveform'].data.cpu().numpy().squeeze()

        else:
            sep_segment = np.zeros(segment_samples)

        sep_audio.append(sep_segment)
        pointer += segment_samples
        segment_index += 1

    sep_audio = np.concatenate(sep_audio, axis=0)

    return sep_audio
'''

def separate_by_query_conditions(audio, segment_samples, at_probs, subclass_indexes, pl_model, device):

    audio_samples = audio.shape[-1]
    pointer = 0
    segment_index = 0
    # segments_num, query_net_bottlenecks_dim = query_net_bottlenecks.shape
    at_threshold = 0.2
    sep_audio = []

    while pointer < audio_samples:

        print(at_probs[segment_index, subclass_indexes].shape)
        # condition = np.zeros(classes_num)
        max_prob = np.max(at_probs[segment_index, subclass_indexes])

        if max_prob < at_threshold:
            sep_segment = np.zeros(segment_samples)

        else:

            segment = librosa.util.fix_length(
                data=audio[pointer : pointer + segment_samples],
                size=segment_samples,
                axis=0,
            )

            segment = torch.Tensor(segment).to(device)[None, :]

            with torch.no_grad():
                pl_model.query_net.eval()

                bottleneck = pl_model.query_net.forward_base(source=segment)
                
                masked_bottleneck = torch.zeros_like(bottleneck)
                masked_bottleneck[:, subclass_indexes] = bottleneck[:, subclass_indexes]

                condition = pl_model.query_net.forward_adaptor(masked_bottleneck)

            input_dict = {
                'mixture': torch.Tensor(segment[:, None, :]),
                'condition': torch.Tensor(condition),
            }

            with torch.no_grad():
                pl_model.ss_model.eval()
                output_dict = pl_model.ss_model(input_dict=input_dict)
            
            sep_segment = output_dict['waveform'].squeeze((0, 1)).data.cpu().numpy()

        sep_audio.append(sep_segment)
        pointer += segment_samples
        segment_index += 1

    sep_audio = np.concatenate(sep_audio, axis=0)

    return sep_audio

'''
def write_audio(audio, output_path, sample_rate, non_sil_threshold=1e-6):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if np.max(audio) > non_sil_threshold:

        soundfile.write(file=output_path, data=audio, samplerate=sample_rate)
        print("Write out to {}".format(output_path))
'''
def write_audio(audio, output_path, sample_rate):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    soundfile.write(file=output_path, data=audio, samplerate=sample_rate)
    print("Write out to {}".format(output_path))

def separate(args):

    audio_path = args.audio_path
    levels = args.levels
    checkpoint_path = args.checkpoint_path
    output_dir = args.output_dir

    config_yaml = "./scripts/train/tmp.yaml"

    configs = read_yaml(config_yaml)
    # at_model_name = configs['query_net']['model_type']
    # model_type = configs['model']['model_type']
    # input_channels = configs['model']['input_channels']
    # output_channels = configs['model']['output_channels']
    # condition_size = configs['data']['condition_size']
    # condition_type = configs['data']['condition_type']
    sample_rate = configs['data']['sample_rate']
    classes_num = configs["data"]["classes_num"]
    segment_samples = sample_rate * 2
    device = 'cuda'
    non_sil_threshold = 1e-6

    # assert condition_type == "at_soft"

    if not output_dir:
        output_dir = os.path.join("separated_results", pathlib.Path(audio_path).stem)

    at_model = load_pretrained_model(
        model_type="Cnn14",
        checkpoint_path="./downloaded_checkpoints/Cnn14_mAP=0.431.pth",
        freeze=True,
    ).to(device)

    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path, 
        device=device,
    )

    # Load audio.
    audio, fs = librosa.load(path=audio_path, sr=sample_rate, mono=True)
    audio_samples = audio.shape[-1]

    at_probs = calculate_segment_at_probs(
        audio=audio, 
        segment_samples=segment_samples, 
        at_model=at_model,
        device=device,
    )
    
    # Parse ontology.
    root = get_ontology_tree()

    nodes = Node.traverse(root)

    for level in levels:

        print("------ Level {} ------".format(level))

        nodes_level_n = get_nodes_with_level_n(nodes=nodes, level=level)

        for node in nodes_level_n:

            class_id = node.class_id

            subclass_indexes = get_children_indexes(node=node)

            sep_audio = separate_by_query_conditions(
                audio=audio, 
                segment_samples=segment_samples, 
                at_probs=at_probs,
                subclass_indexes=subclass_indexes, 
                pl_model=pl_model,
                device=device,
            )

            sep_audio = sep_audio[0 : audio_samples]

            # Write out separated audio.
            label = audioset632_id_to_lb[class_id]
            output_path = os.path.join(output_dir, "level={}".format(level), "{}.wav".format(label))

            if np.max(audio) > non_sil_threshold:
                write_audio(
                    audio=sep_audio,
                    output_path=output_path,
                    sample_rate=sample_rate,
                )
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str)
    parser.add_argument('--levels', nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    separate(args)