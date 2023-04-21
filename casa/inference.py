import os
import re
import librosa
import time
import pathlib
import pickle
import json
import soundfile
import argparse
import matplotlib.pyplot as plt
import lightning.pytorch as pl

from casa.utils import calculate_sdr
from casa.utils import create_logging, parse_yaml, load_pretrained_panns

from casa.data.anchor_segment_detectors import AnchorSegmentDetector
from casa.data.anchor_segment_mixers import AnchorSegmentMixer
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


def separate(args) -> None:
    r"""Do separation of active sound classes."""

    # Arguments & parameters
    audio_path = args.audio_path
    levels = args.levels
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    output_dir = args.output_dir
    
    non_sil_threshold = 1e-6
    device = 'cuda'
    ontology_path = './metadata/ontology.json'

    configs = parse_yaml(config_yaml)
    sample_rate = configs['data']['sample_rate']
    classes_num = configs["data"]["classes_num"]
    segment_seconds = configs["data"]["segment_seconds"]
    segment_samples = int(sample_rate * segment_seconds)
    
    # Create directory
    if not output_dir:
        output_dir = os.path.join("separated_results", pathlib.Path(audio_path).stem)

    # Load pretrained universal source separation model
    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path, 
    ).to(device)

    # Load audio
    audio, fs = librosa.load(path=audio_path, sr=sample_rate, mono=True)
    audio_samples = audio.shape[-1]

    # Load pretrained audio tagging model
    at_model = load_pretrained_panns(
        model_type="Cnn14",
        checkpoint_path="./downloaded_checkpoints/Cnn14_mAP=0.431.pth",
        freeze=True,
    ).to(device)

    at_probs = calculate_segment_at_probs(
        audio=audio, 
        segment_samples=segment_samples, 
        at_model=at_model,
        device=device,
    )
    # at_probs: (segments_num, condition_dim)
    
    # Parse and build AudioSet ontology tree
    root = get_ontology_tree(ontology_path=ontology_path)

    nodes = Node.traverse(root)

    for level in levels:

        print("------ Level {} ------".format(level))

        nodes_level_n = get_nodes_with_level_n(nodes=nodes, level=level)

        hierarchy_at_probs = []

        for node in nodes_level_n:

            class_id = node.class_id

            subclass_indexes = get_children_indexes(node=node)
            # E.g., [0, 1, ..., 71]
            
            if len(subclass_indexes) == 0:
                continue

            sep_audio = separate_by_query_conditions(
                audio=audio, 
                segment_samples=segment_samples, 
                at_probs=at_probs,
                subclass_indexes=subclass_indexes, 
                pl_model=pl_model,
                device=device,
            )
            # sep_audio: (audio_samples,)

            # Write out separated audio
            label = audioset632_id_to_lb[class_id]
            output_path = os.path.join(output_dir, "level={}".format(level), "{}.wav".format(label))

            if np.max(sep_audio) > non_sil_threshold:
                write_audio(
                    audio=sep_audio,
                    output_path=output_path,
                    sample_rate=sample_rate,
                )

            hierarchy_at_prob = np.max(at_probs[:, subclass_indexes], axis=-1)
            hierarchy_at_probs.append(hierarchy_at_prob)

        hierarchy_at_probs = np.stack(hierarchy_at_probs, axis=-1)
        plt.matshow(hierarchy_at_probs.T, origin='lower', aspect='auto', cmap='jet')
        plt.savefig('_zz_{}.pdf'.format(level))


def load_ss_model(
    configs: Dict, 
    checkpoint_path: str, 
) -> nn.Module:
    r"""Load trained universal source separation model.

    Args:
        configs (Dict)
        checkpoint_path (str): path of the checkpoint to load
        device (str): e.g., "cpu" | "cuda"
    """
    
    # Initialize query net
    query_net = initialize_query_net(
        configs=configs,
    )

    ss_model_type = configs["ss_model"]["model_type"]
    input_channels = configs["ss_model"]["input_channels"]
    output_channels = configs["ss_model"]["output_channels"]
    condition_size = configs["query_net"]["outputs_num"]

    # Initialize separation model
    SsModel = get_model_class(model_type=ss_model_type)

    ss_model = SsModel(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
    )

    # Load PyTorch Lightning model
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
    )

    return pl_model


def calculate_segment_at_probs(
    audio: np.ndarray, 
    segment_samples: int, 
    at_model: nn.Module, 
    device: str,
) -> np.ndarray:
    r"""Split audio into short segments. Calcualte the audio tagging 
    predictions of all segments.
    
    Args:
        audio (np.ndarray): (audio_samples,)
        segment_samples (int): short segment duration
        at_model (nn.Module): pretrained audio tagging model
        device (str): "cpu" | "cuda"

    Returns:
        at_probs (np.ndarray): audio tagging probabilities on all segments, 
            (segments_num, classes_num)
    """
    
    audio_samples = audio.shape[-1]
    pointer = 0
    at_probs = []

    while pointer < audio_samples:

        segment = librosa.util.fix_length(
            data=audio[pointer : pointer + segment_samples],
            size=segment_samples,
            axis=0,
        )
        
        segments = torch.Tensor(segment).unsqueeze(dim=0).to(device)
        # segments: (batch_size=1, segment_samples)

        with torch.no_grad():
            at_model.eval()
            at_prob = at_model(input=segments)['clipwise_output']

        at_prob = at_prob.squeeze(dim=0).data.cpu().numpy()
        # at_prob: (classes_num,)

        at_probs.append(at_prob)

        pointer += segment_samples

    at_probs = np.stack(at_probs, axis=0)
    # at_probs: (segments_num, condition_dim)
    
    return at_probs


# def get_nodes_with_level_n(nodes: List[Node], level: int) -> list[Node]:
def get_nodes_with_level_n(nodes, level):
    r"""Return nodes with level=N."""

    nodes_level_n = []

    for node in nodes:
        if node.level == level:
            nodes_level_n.append(node)

    return nodes_level_n


def get_children_indexes(node: Node) -> List[int]:
    r"""Get class indexes of all children of a node."""

    nodes_level_n_children = Node.traverse(node=node)

    subclass_indexes = [ID_TO_IX[node.class_id] for node in nodes_level_n_children if node.class_id in ID_TO_IX]

    return subclass_indexes

'''
def separate_by_query_conditions(
    audio: np.ndarray, 
    segment_samples: int, 
    at_probs: np.ndarray, 
    subclass_indexes: List[int], 
    pl_model: pl.LightningModule, 
    device: str,
):
    r"""Do separation for active segments.

    Args:
        audio (np.ndarray): audio clip
        segment_samples (int): segment samples
        at_probs (np.ndarray): predicted audio tagging probability on segments, 
            (segments_num, classes_num)
        subclass_indexes (List[int]): all values in subclass_indexes are 
            remained to build the condition
        pl_model (pl.LightningModule): trained universal source separation model
        device (str), e.g., "cpu" | "cuda"
    """

    audio_samples = audio.shape[-1]
    pointer = 0
    segment_index = 0
    at_threshold = 0.2
    sep_audio = []

    while pointer < audio_samples:

        max_prob = np.max(at_probs[segment_index, subclass_indexes])

        if max_prob < at_threshold:
            # Set separation results to silence for non-active segments
            sep_segment = np.zeros(segment_samples)

        else:
            # Only do separation for active segments
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

def separate_by_query_conditions(
    audio: np.ndarray, 
    segment_samples: int, 
    at_probs: np.ndarray, 
    subclass_indexes: List[int], 
    pl_model: pl.LightningModule, 
    device: str,
) -> np.ndarray:
    r"""Do separation for active segments depending on the subclass_indexes.

    Args:
        audio (np.ndarray): audio clip
        segment_samples (int): segment samples
        at_probs (np.ndarray): predicted audio tagging probability on segments, 
            (segments_num, classes_num)
        subclass_indexes (List[int]): all values in subclass_indexes are 
            remained to build the condition
        pl_model (pl.LightningModule): trained universal source separation model
        device (str), e.g., "cpu" | "cuda"

    Returns:
        sep_audio (np.ndarray): separated audio
    """

    audio_samples = audio.shape[-1]
    at_threshold = 0.2
    batch_size = 8

    segments_num = int(np.ceil(audio_samples / segment_samples))

    active_segment_indexes = []
    active_segments = []

    # Collect active segments
    for segment_index in range(segments_num):

        max_prob = np.max(at_probs[segment_index, subclass_indexes])

        if max_prob >= at_threshold:
            # Only do separation for active segments

            begin_sample = segment_index * segment_samples
            end_sample = begin_sample + segment_samples

            segment = librosa.util.fix_length(
                data=audio[begin_sample : end_sample],
                size=segment_samples,
                axis=0,
            )

            active_segments.append(segment)
            active_segment_indexes.append(segment_index)

    if len(active_segments) > 0:
        active_segments = np.stack(active_segments, axis=0)
        active_segment_indexes = np.stack(active_segment_indexes, axis=0)

    # Do separation in mini-batch
    pointer = 0
    active_sep_segments = []

    while pointer < len(active_segments):

        batch_segments = torch.Tensor(active_segments[pointer : pointer + batch_size]).to(device)

        batch_sep_segments = _do_sep_in_minibatch(
            batch_segments=batch_segments, 
            subclass_indexes=subclass_indexes,
            pl_model=pl_model, 
            device=device
        )
        active_sep_segments.extend(batch_sep_segments)
        pointer += batch_size

    # Get separated segments
    sep_segments = np.zeros((segments_num, segment_samples))

    for i in range(len(active_segment_indexes)):
        sep_segments[active_segment_indexes[i]] = active_sep_segments[i]

    sep_audio = sep_segments.flatten()[0 : audio_samples]
    
    return sep_audio


def _do_sep_in_minibatch(
    batch_segments: np.ndarray, 
    subclass_indexes: List[int], 
    pl_model: pl.LightningModule, 
    device: str,
) -> np.ndarray:
    r"""Separate by mini-batch.

    Args:
        batch_segments (np.ndarray): (batch_size, segment_samples)
        subclass_indexes (List[int]): a list of subclasses
        pl_model (pl.LightningModule): universal separation model
        device (str): e.g., "cpu" | "cuda"

    Returns:
        batch_sep_segments (np.ndarray): separated mini-batch segments
    """

    batch_segments = torch.Tensor(batch_segments).to(device)
    # shape: (batch_size, segment_samples)

    with torch.no_grad():
        pl_model.query_net.eval()

        bottleneck = pl_model.query_net.forward_base(source=batch_segments)
        # bottleneck: (batch_size, bottleneck_dim)
        
        masked_bottleneck = torch.zeros_like(bottleneck)
        masked_bottleneck[:, subclass_indexes] = bottleneck[:, subclass_indexes]

        condition = pl_model.query_net.forward_adaptor(masked_bottleneck)
        # condition: (batch_size, condition_dim)

    input_dict = {
        'mixture': torch.Tensor(batch_segments.unsqueeze(1)),
        'condition': torch.Tensor(condition),
    }

    with torch.no_grad():
        pl_model.ss_model.eval()
        output_dict = pl_model.ss_model(input_dict=input_dict)
    
    batch_sep_segments = output_dict['waveform'].squeeze(dim=1).data.cpu().numpy()
    # (batch_size, segment_samples)

    return batch_sep_segments


def write_audio(
    audio: np.ndarray, 
    output_path: str, 
    sample_rate: int,
):
    r"""Write audio to disk.

    Args:
        audio (np.ndarray): audio to write out
        output_path (str): path to write out the audio
        sample_rate (int)

    Returns:
        None
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    soundfile.write(file=output_path, data=audio, samplerate=sample_rate)
    print("Write out to {}".format(output_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str)
    parser.add_argument('--levels', nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument('--config_yaml', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    separate(args)