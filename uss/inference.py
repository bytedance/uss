import argparse
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List

import librosa
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
import torch.nn as nn

from uss.config import (ID_TO_IX, IX_TO_LB, LB_TO_IX, csv_paths_dict,
                        panns_paths_dict)
from uss.models.pl_modules import LitSeparation, get_model_class
from uss.models.query_nets import initialize_query_net
from uss.parse_ontology import Node, get_ontology_tree
from uss.utils import (get_audioset632_id_to_lb, get_path,
                       load_pretrained_panns, parse_yaml, remove_silence,
                       repeat_to_length)


def separate(args) -> None:
    r"""Do separation for active sound classes."""

    # Arguments & parameters
    audio_path = args.audio_path
    levels = args.levels
    class_ids = args.class_ids
    queries_dir = args.queries_dir
    query_emb_path = args.query_emb_path
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    output_dir = args.output_dir

    non_sil_threshold = 1e-6
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ontology_path = get_path(csv_paths_dict["ontology.csv"])

    configs = parse_yaml(config_yaml)
    sample_rate = configs["data"]["sample_rate"]
    segment_seconds = configs["data"]["segment_seconds"]
    segment_samples = int(sample_rate * segment_seconds)

    print("Using {}.".format(device))

    # Create directory
    if not output_dir:
        output_dir = os.path.join(
            "separated_results",
            Path(audio_path).stem)

    # Load pretrained universal source separation model
    print("Loading model ...")

    warnings.filterwarnings("ignore", category=UserWarning)

    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
    ).to(device)

    # Load audio
    audio, fs = librosa.load(path=audio_path, sr=sample_rate, mono=True)

    # Load pretrained audio tagging model
    at_model_type = "Cnn14"

    at_model = load_pretrained_panns(
        model_type=at_model_type,
        checkpoint_path=get_path(panns_paths_dict[at_model_type]),
        freeze=True,
    ).to(device)

    flag_sum = sum([
        len(levels) > 0,
        len(class_ids) > 0,
        len(queries_dir) > 0,
        len(query_emb_path) > 0,
    ])

    assert flag_sum in [0, 1], "Please use only `levels` or `class_ids` or \
        `queries_dir` or `query_emb_path` arguments."

    if flag_sum == 0:
        levels = [1, 2, 3]

    print("Separating ...")

    # Separate by hierarchy
    if len(levels) > 0:
        separate_by_hierarchy(
            audio=audio,
            sample_rate=sample_rate,
            segment_samples=segment_samples,
            at_model=at_model,
            pl_model=pl_model,
            device=device,
            levels=levels,
            ontology_path=ontology_path,
            non_sil_threshold=non_sil_threshold,
            output_dir=output_dir
        )

    # Separate by class IDs
    elif len(class_ids) > 0:
        separate_by_class_ids(
            audio=audio,
            sample_rate=sample_rate,
            segment_samples=segment_samples,
            at_model=at_model,
            pl_model=pl_model,
            device=device,
            class_ids=class_ids,
            output_dir=output_dir
        )

    # Calculate query embedding and do separation
    elif len(queries_dir) > 0:

        print("Calculate query condition ...")
        query_time = time.time()

        query_condition = calculate_query_emb(
            queries_dir=queries_dir,
            pl_model=pl_model,
            sample_rate=sample_rate,
            remove_sil=True,
            segment_samples=segment_samples,
        )

        print("Time: {:.3f} s".format(time.time() - query_time))

        pickle_path = os.path.join(
            "./query_conditions",
            "config={}".format(
                Path(config_yaml).stem),
            "{}.pkl".format(
                Path(queries_dir).stem))

        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

        pickle.dump(query_condition, open(pickle_path, 'wb'))
        print("Write query condition to {}".format(pickle_path))

        output_path = os.path.join(
            output_dir, "query={}.wav".format(
                Path(queries_dir).stem))

        separate_by_query_condition(
            audio=audio,
            segment_samples=segment_samples,
            sample_rate=sample_rate,
            query_condition=query_condition,
            pl_model=pl_model,
            output_path=output_path,
        )

    # Load pre-calculated query embedding and do separation
    elif Path(query_emb_path).is_file():

        query_condition = pickle.load(open(query_emb_path, 'rb'))

        output_path = os.path.join(
            output_dir, "query={}.wav".format(
                Path('111').stem))

        separate_by_query_condition(
            audio=audio,
            segment_samples=segment_samples,
            sample_rate=sample_rate,
            query_condition=query_condition,
            pl_model=pl_model,
            output_path=output_path,
        )


def load_ss_model(
    configs: Dict,
    checkpoint_path: str,
) -> nn.Module:
    r"""Load trained universal source separation model.

    Args:
        configs (Dict)
        checkpoint_path (str): path of the checkpoint to load
        device (str): e.g., "cpu" | "cuda"

    Returns:
        pl_model: pl.LightningModule
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
        map_location="cpu",
    )

    return pl_model


def separate_by_hierarchy(
    audio: np.ndarray,
    sample_rate: int,
    segment_samples: int,
    at_model: nn.Module,
    pl_model: pl.LightningModule,
    device: str,
    levels: List[int],
    ontology_path: str,
    non_sil_threshold: float,
    output_dir: str,
) -> None:
    r"""Separate by hierarchy."""

    audioset632_id_to_lb = get_audioset632_id_to_lb(
        ontology_path=ontology_path)

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

            output_name = "{}.wav".format(label)

            # if label in LB_TO_IX.keys():
            #     output_name = "classid={}_{}.wav".format(LB_TO_IX[label], label)
            # else:
            #     output_name = "classid=unknown_{}.wav".format(label)

            output_path = os.path.join(
                output_dir,
                "level={}".format(level),
                output_name,
            )

            if np.max(sep_audio) > non_sil_threshold:
                write_audio(
                    audio=sep_audio,
                    output_path=output_path,
                    sample_rate=sample_rate,
                )

            hierarchy_at_prob = np.max(at_probs[:, subclass_indexes], axis=-1)
            hierarchy_at_probs.append(hierarchy_at_prob)

        hierarchy_at_probs = np.stack(hierarchy_at_probs, axis=-1)
        plt.matshow(
            hierarchy_at_probs.T,
            origin="lower",
            aspect="auto",
            cmap="jet")
        plt.savefig("_zz_{}.pdf".format(level))


def separate_by_class_ids(
    audio: np.ndarray,
    sample_rate: int,
    segment_samples: int,
    at_model: nn.Module,
    pl_model: pl.LightningModule,
    device: str,
    class_ids: List[int],
    output_dir: str,
) -> None:
    r"""Separate by class IDs."""

    at_probs = calculate_segment_at_probs(
        audio=audio,
        segment_samples=segment_samples,
        at_model=at_model,
        device=device,
    )
    # at_probs: (segments_num, condition_dim)

    sep_audio = separate_by_query_conditions(
        audio=audio,
        segment_samples=segment_samples,
        at_probs=at_probs,
        subclass_indexes=class_ids,
        pl_model=pl_model,
        device=device,
    )
    # sep_audio: (audio_samples,)

    # Write out separated audio
    output_name = ";".join(
        ["{}_{}".format(class_id, IX_TO_LB[class_id]) for class_id in class_ids])
    output_name += ".wav"

    output_path = os.path.join(
        output_dir,
        output_name,
    )

    write_audio(
        audio=sep_audio,
        output_path=output_path,
        sample_rate=sample_rate,
    )


def calculate_query_emb(
    queries_dir: str,
    pl_model: pl.LightningModule,
    sample_rate: int,
    remove_sil: bool,
    segment_samples: int,
    batch_size=8,
) -> np.ndarray:
    r"""Calculate the query embddings of audio files in a directory."""

    audio_names = sorted(os.listdir(queries_dir))

    avg_query_conditions = []

    # Average query conditions of all audios
    for audio_index, audio_name in enumerate(audio_names):

        print("{} / {}, {}".format(audio_index, len(audio_names), audio_name))

        audio_path = os.path.join(queries_dir, audio_name)

        audio, fs = librosa.load(path=audio_path, sr=sample_rate, mono=True)

        # Remove silence
        if remove_sil:
            audio = remove_silence(audio=audio, sample_rate=sample_rate)

        audio_samples = audio.shape[0]

        segments_num = int(np.ceil(audio_samples / segment_samples))

        segments = []

        # Get all segments
        for segment_index in range(segments_num):

            begin_sample = segment_index * segment_samples
            end_sample = begin_sample + segment_samples

            segment = audio[begin_sample: end_sample]
            segment = repeat_to_length(
                audio=segment, segment_samples=segment_samples)
            segments.append(segment)

        if len(segments) == 0:
            continue

        segments = np.stack(segments, axis=0)

        # Calcualte query conditions in mini-batch
        pointer = 0
        query_conditions = []

        while pointer < len(segments):

            batch_segments = segments[pointer: pointer + batch_size]

            query_condition = _do_query_in_minibatch(
                batch_segments=batch_segments,
                query_net=pl_model.query_net,
            )

            query_conditions.extend(query_condition)
            pointer += batch_size

        avg_query_condition = np.mean(query_conditions, axis=0)
        avg_query_conditions.append(avg_query_condition)

    # Average query conditions of all audio files
    avg_query_condition = np.mean(avg_query_conditions, axis=0)

    return avg_query_condition


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
            data=audio[pointer: pointer + segment_samples],
            size=segment_samples,
            axis=0,
        )

        segments = torch.Tensor(segment).unsqueeze(dim=0).to(device)
        # segments: (batch_size=1, segment_samples)

        with torch.no_grad():
            at_model.eval()
            at_prob = at_model(input=segments)["clipwise_output"]

        at_prob = at_prob.squeeze(dim=0).data.cpu().numpy()
        # at_prob: (classes_num,)

        at_probs.append(at_prob)

        pointer += segment_samples

    at_probs = np.stack(at_probs, axis=0)
    # at_probs: (segments_num, condition_dim)

    return at_probs


def get_nodes_with_level_n(nodes: List[Node], level: int) -> List[Node]:
    r"""Return nodes with level=N."""

    nodes_level_n = []

    for node in nodes:
        if node.level == level:
            nodes_level_n.append(node)

    return nodes_level_n


def get_children_indexes(node: Node) -> List[int]:
    r"""Get class indexes of all children of a node."""

    nodes_level_n_children = Node.traverse(node=node)

    subclass_indexes = [ID_TO_IX[node.class_id]
                        for node in nodes_level_n_children if node.class_id in ID_TO_IX]

    return subclass_indexes


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
                data=audio[begin_sample: end_sample],
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

        batch_segments = active_segments[pointer: pointer + batch_size]

        batch_sep_segments = _do_sep_by_id_in_minibatch(
            batch_segments=batch_segments,
            subclass_indexes=subclass_indexes,
            pl_model=pl_model,
        )
        active_sep_segments.extend(batch_sep_segments)
        pointer += batch_size

    # Get separated segments
    sep_segments = np.zeros((segments_num, segment_samples))

    for i in range(len(active_segment_indexes)):
        sep_segments[active_segment_indexes[i]] = active_sep_segments[i]

    sep_audio = sep_segments.flatten()[0: audio_samples]

    return sep_audio


def separate_by_query_condition(
    audio: np.ndarray,
    segment_samples: int,
    sample_rate: int,
    query_condition: np.ndarray,
    pl_model: pl.LightningModule,
    output_path: str,
    batch_size: int = 8,
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

    segments_num = int(np.ceil(audio_samples / segment_samples))

    segments = []

    # Collect active segments
    for segment_index in range(segments_num):

        begin_sample = segment_index * segment_samples
        end_sample = begin_sample + segment_samples

        segment = librosa.util.fix_length(
            data=audio[begin_sample: end_sample],
            size=segment_samples,
            axis=0,
        )

        segments.append(segment)

    segments = np.stack(segments, axis=0)

    # Do separation in mini-batch
    pointer = 0
    sep_segments = []

    while pointer < len(segments):

        batch_segments = segments[pointer: pointer + batch_size]

        batch_sep_segments = _do_sep_by_query_in_minibatch(
            batch_segments=batch_segments,
            query_condition=query_condition,
            ss_model=pl_model.ss_model,
        )
        sep_segments.extend(batch_sep_segments)
        pointer += batch_size

    sep_segments = np.concatenate(sep_segments, axis=0)

    sep_audio = sep_segments.flatten()[0: audio_samples]

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        soundfile.write(
            file=output_path,
            data=sep_audio,
            samplerate=sample_rate)
        print("Write out separated file to {}".format(output_path))

    return sep_audio


def _do_sep_by_id_in_minibatch(
    batch_segments: np.ndarray,
    subclass_indexes: List[int],
    pl_model: pl.LightningModule,
) -> np.ndarray:
    r"""Separate by class IDs in mini-batch.

    Args:
        batch_segments (np.ndarray): (batch_size, segment_samples)
        subclass_indexes (List[int]): a list of subclasses
        pl_model (pl.LightningModule): universal separation model

    Returns:
        batch_sep_segments (np.ndarray): separated mini-batch segments
    """

    device = pl_model.device

    batch_segments = torch.Tensor(batch_segments).to(device)
    # shape: (batch_size, segment_samples)

    with torch.no_grad():
        pl_model.query_net.eval()

        bottleneck = pl_model.query_net.forward_base(source=batch_segments)
        # bottleneck: (batch_size, bottleneck_dim)

        masked_bottleneck = torch.zeros_like(bottleneck)
        masked_bottleneck[:,
                          subclass_indexes] = bottleneck[:,
                                                         subclass_indexes]

        condition = pl_model.query_net.forward_adaptor(masked_bottleneck)
        # condition: (batch_size, condition_dim)

    input_dict = {
        "mixture": torch.Tensor(batch_segments.unsqueeze(1)),
        "condition": torch.Tensor(condition),
    }

    with torch.no_grad():
        pl_model.ss_model.eval()
        output_dict = pl_model.ss_model(input_dict=input_dict)

    batch_sep_segments = output_dict["waveform"].squeeze(
        dim=1).data.cpu().numpy()
    # (batch_size, segment_samples)

    return batch_sep_segments


def _do_sep_by_query_in_minibatch(
    batch_segments: np.ndarray,
    query_condition: np.ndarray,
    ss_model: nn.Module,
) -> np.ndarray:
    r"""Separate by query condition in mini-batch.

    Args:
        batch_segments (np.ndarray): (batch_size, segment_samples)
        query_condition (np.ndarray): (batch_size, embedding_dim)
        pl_model (pl.LightningModule): universal separation model

    Returns:
        batch_sep_segments (np.ndarray): separated mini-batch segments
    """

    device = next(ss_model.parameters()).device

    batch_segments = torch.Tensor(batch_segments).to(device).unsqueeze(dim=1)
    # shape: (batch_size, 1, segment_samples)

    query_condition = torch.Tensor(query_condition).to(device).unsqueeze(dim=0)

    input_dict = {
        "mixture": batch_segments,
        "condition": query_condition,
    }

    with torch.no_grad():
        ss_model.eval()
        output_dict = ss_model(input_dict=input_dict)

    batch_sep_segments = output_dict["waveform"].squeeze(
        dim=1).data.cpu().numpy()
    # (batch_size, segment_samples)

    return batch_sep_segments


def _do_query_in_minibatch(
    batch_segments: np.ndarray,
    query_net: nn.Module,
) -> np.ndarray:
    r"""Separate by mini-batch.

    Args:
        batch_segments (np.ndarray): (batch_size, segment_samples)
        pl_model (pl.LightningModule): universal separation model

    Returns:
        batch_condition (np.ndarray): mini-batch conditions.
    """

    device = next(query_net.parameters()).device

    batch_segments = torch.Tensor(batch_segments).to(device)
    # shape: (batch_size, segment_samples)

    with torch.no_grad():
        query_net.eval()

        batch_condition = query_net.forward(source=batch_segments)["output"]
        # condition: (batch_size, condition_dim)

    batch_condition = batch_condition.data.cpu().numpy()

    return batch_condition


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--levels", nargs="*", type=int, default=[])
    parser.add_argument("--class_ids", nargs="*", type=int, default=[])
    parser.add_argument("--queries_dir", type=str, default="")
    parser.add_argument("--query_emb_path", type=str, default="")
    parser.add_argument("--config_yaml", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    separate(args)
