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
from casa.utils import create_logging, read_yaml, load_pretrained_model #, load_pretrained_sed_model, 

from casa.data.anchor_segment_detectors import AnchorSegmentDetector
from casa.data.anchor_segment_mixers import AnchorSegmentMixer
from casa.data.query_condition_extractors import QueryConditionExtractor
from casa.models.pl_modules import LitSeparation
from casa.models.resunet import *
from casa.config import IX_TO_LB
from casa.parse_ontology import get_ontology_tree, get_subclass_indexes


ontology_path = 'metadata/ontology.json'
audioset632_id_to_lb = {}
with open(ontology_path) as f:
    data_list = json.load(f)
for e in data_list:
    audioset632_id_to_lb[e['id']] = e['name']


def load_ss_model(at_model_name, ss_model_name, input_channels, output_channels, condition_type, condition_size, checkpoint_path, device):

    at_model = load_pretrained_model(
        model_name=at_model_name,
        checkpoint_path=None,
        freeze=None,
    )

    query_condition_extractor = QueryConditionExtractor(
        at_model=at_model,
        condition_type=condition_type,
    )

    Model = eval(ss_model_name)
    # Model = str_to_class(model_type)

    ss_model = Model(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
    )

    # step = 300000
    # checkpoint_path = "/home/tiger/workspaces/casa/checkpoints/train/config={},devices=1/step={}.ckpt".format(pathlib.Path(config_yaml).stem, step)

    pl_model = LitSeparation.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False,
        ss_model=ss_model,
        anchor_segment_detector=None,
        anchor_segment_mixer=None,
        query_condition_extractor=query_condition_extractor,
        loss_function=None,
        learning_rate=None,
        lr_lambda=None,
    ).to(device)

    return pl_model


def add(args):

    audio_path = args.audio_path

    config_yaml = "./scripts/train/01b.yaml"

    configs = read_yaml(config_yaml)
    at_model_name = configs['audio_tagging']['model_name']
    model_type = configs['model']['model_type']
    input_channels = configs['model']['input_channels']
    output_channels = configs['model']['output_channels']
    condition_size = configs['data']['condition_size']
    condition_type = configs['data']['condition_type']
    sample_rate = configs['data']['sample_rate']
    classes_num = configs["data"]["classes_num"]
    segment_samples = sample_rate * 2
    device = 'cuda'

    assert condition_type == "at_soft"

    step = 300000
    checkpoint_path = "/home/tiger/workspaces/casa/checkpoints/train/config={},devices=1/step={}.ckpt".format(pathlib.Path(config_yaml).stem, step)

    pl_model = load_ss_model(
        at_model_name=at_model_name, 
        ss_model_name=model_type, 
        input_channels=input_channels, 
        output_channels=output_channels, 
        condition_type=condition_type, 
        condition_size=condition_size, 
        checkpoint_path=checkpoint_path, 
        device=device,
    )

    # audio_path = "./resources/harry_potter.flac"
    audio, fs = librosa.load(audio_path, sr=sample_rate, mono=True)

    ###
    # cluster_levels = [3]
    hierarchies = [1, 2]

    # Parse ontology.
    root = get_ontology_tree(verbose=False)

    # Get id_list of all sound classes with the target cluster_levels.
    class_id_list = []
    nodes = Node.traverse(root)

    for node, layer in zip(nodes, layers):
        if layer in cluster_levels:
            if len(get_subclass_indexes(root, id=node.id)) > 0:
                id_list.append(node.id)

    id_indexes = {}


    # for _n, target_id in enumerate(id_list):
    for k, class_id in enumerate(id_list):

        subclass_indexes = get_subclass_indexes(root, id=class_id)

        pointer = 0
        sep_audio = []
        # at_probs = []
        # sed_probs = []

        while pointer < audio.shape[-1]:

            segment = librosa.util.fix_length(
                data=audio[pointer : pointer + segment_samples],
                size=segment_samples,
                axis=0,
            )
            
            segments = torch.Tensor(segment)[None, :].to(device)

            with torch.no_grad():
                pl_model.query_condition_extractor.eval()

                query_condition = pl_model.query_condition_extractor(
                    segments=segments,
                ).squeeze().data.cpu().numpy()

            diff_indexes = np.setdiff1d(ar1=np.arange(classes_num), ar2=subclass_indexes)
            query_condition[diff_indexes] = 0

            if np.max(query_condition > 0.2):
                input_dict = {
                    'mixture': segments[:, None, :],
                    'condition': torch.Tensor(query_condition[None, :]).to(device),
                }

                # from IPython import embed; embed(using=False); os._exit(0)
                with torch.no_grad():
                    pl_model.ss_model.eval()
                    output_dict = pl_model.ss_model(input_dict=input_dict)
                    
                sep_segment = output_dict['waveform'].data.cpu().numpy().squeeze()
            else:
                sep_segment = np.zeros(segment_samples)

            sep_audio.append(sep_segment)
            pointer += segment_samples

        sep_audio = np.concatenate(sep_audio, axis=0)

        if np.max(sep_audio) > 1e-6:
            
            label = audioset632_id_to_lb[class_id]

            output_path = "_tmp2/level={}/{}.wav".format(cluster_levels[0], label)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            soundfile.write(file=output_path, data=sep_audio, samplerate=sample_rate)
            print("Write out to {}".format(output_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str)

    args = parser.parse_args()

    add(args)