# import os
# import re
# import librosa
# import time

# from uss.utils import calculate_sdr
# from uss.utils import create_logging, read_yaml, load_pretrained_model #, load_pretrained_sed_model, 

# from uss.data.anchor_segment_detectors import AnchorSegmentDetector
# from uss.data.anchor_segment_mixers import AnchorSegmentMixer
# from uss.data.query_condition_extractors import QueryConditionExtractor
# from uss.models.pl_modules import LitSeparation
# from uss.models.resunet import *

# from torch.utils.tensorboard import SummaryWriter
import torch
from pathlib import Path


class AudiosetEvaluator:
    def __init__(self, pl_model, audios_dir, classes_num, max_eval_per_class=None):

        self.pl_model = pl_model
        self.audios_dir = audios_dir
        self.classes_num = classes_num
        self.device = pl_model.device
        self.max_eval_per_class = max_eval_per_class

    @torch.no_grad()
    def __call__(self):

        sdrs_dict = {class_id: [] for class_id in range(self.classes_num)}
        sdris_dict = {class_id: [] for class_id in range(self.classes_num)}

        for class_id in range(self.classes_num):

            sub_dir = os.path.join(self.audios_dir, "classid={}".format(class_id))            

            audio_names = self._get_audio_names(audios_dir=sub_dir)

            for audio_index, audio_name in enumerate(audio_names):

                source_path = os.path.join(sub_dir, "{},source.wav".format(audio_name))
                mixture_path = os.path.join(sub_dir, "{},mixture.wav".format(audio_name))

                source, fs = librosa.load(source_path, sr=None, mono=True)
                mixture, fs = librosa.load(mixture_path, sr=None, mono=True)

                sdr_no_sep = calculate_sdr(ref=source, est=mixture)
                
                conditions = self.pl_model.query_condition_extractor(
                    segments=torch.Tensor(source)[None, :].to(self.device),
                )
                
                input_dict = {
                    'mixture': torch.Tensor(mixture)[None, None, :].to(self.device),
                    'condition': conditions,
                }

                self.pl_model.eval()
                sep_segment = self.pl_model.ss_model(input_dict)['waveform']
                sep_segment = sep_segment.squeeze().data.cpu().numpy()

                sdr = calculate_sdr(ref=source, est=sep_segment)
                sdri = sdr - sdr_no_sep

                sdrs_dict[class_id].append(sdr)
                sdris_dict[class_id].append(sdri)

                if audio_index == self.max_eval_per_class:
                    break

            print("Class ID: {} / {}, SDR: {:.3f}, SDRi: {:.3f}".format(class_id, self.classes_num, np.mean(sdrs_dict[class_id]), np.mean(sdris_dict[class_id])))

        stats_dict = {
            "sdrs_dict": sdrs_dict,
            "sdris_dict": sdris_dict,
        }

        return stats_dict

    def _get_audio_names(self, audios_dir):
            
        audio_names = sorted(os.listdir(audios_dir))

        audio_names = [re.search('(.*),(mixture|source).wav', audio_name).group(1) for audio_name in audio_names]

        audio_names = sorted(list(set(audio_names)))

        return audio_names


def add2():

    config_yaml = "./scripts/train/01.yaml"
    sample_rate = 32000
    classes_num = 527
    device = 'cuda'

    configs = read_yaml(config_yaml)

    num_workers = configs['train']['num_workers']
    model_type = configs['model']['model_type']
    input_channels = configs['model']['input_channels']
    output_channels = configs['model']['output_channels']
    condition_size = configs['data']['condition_size']
    loss_type = configs['train']['loss_type']
    learning_rate = float(configs['train']['learning_rate'])
    condition_type = configs['data']['condition_type']

    sample_rate = configs['data']['sample_rate']

    at_model = load_pretrained_model(
        model_name=configs['audio_tagging']['model_name'],
        checkpoint_path=configs['audio_tagging']['checkpoint_path'],
        freeze=configs['audio_tagging']['freeze'],
    )

    query_condition_extractor = QueryConditionExtractor(
        at_model=at_model,
        condition_type='embedding',
    )

    Model = eval(model_type)
    # Model = str_to_class(model_type)

    ss_model = Model(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
    )

    # pytorch-lightning model
    pl_model = LitSeparation(
        ss_model=ss_model,
        anchor_segment_detector=None,
        anchor_segment_mixer=None,
        query_condition_extractor=query_condition_extractor,
        loss_function=None,
        learning_rate=None,
        lr_lambda=None,
    )

    checkpoint_path = "/home/tiger/my_code_2019.12-/python/audioset_source_separation/lightning_logs/version_0/checkpoints/lightning_logs/version_8/checkpoints/step=1.ckpt"

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

    audios_dir = "/home/tiger/workspaces/uss/evaluation/audioset/mixtures_sources_test"

    evaluator = AudiosetEvaluator(pl_model=pl_model, audios_dir=audios_dir, classes_num=classes_num, max_eval_per_class=5)

    evaluator()


def add3():
    
    writer = SummaryWriter(log_dir="_tmp/exp1")

    for i in range(10):
        writer.add_scalar('Loss/train', global_step=i, scalar_value=i * 2)
        writer.add_scalar('Loss/test', global_step=i, scalar_value=i * 3)
        


def add4(): 
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator('/home/tiger/my_code_2019.12-/python/audioset_source_separation/workspaces/uss/tf_logs/train/config=01a,devices=1/events.out.tfevents.1681729002.n130-020-141.1680684.0')

    ea.Reload()
    tags = ea.Tags()
    # values = ea.Scalars('SDRi/test')

    from IPython import embed; embed(using=False); os._exit(0)
    

from typing import List, NoReturn
def sub() -> NoReturn:
    raise Exception


def add5():
    a1 = sub()
    from IPython import embed; embed(using=False); os._exit(0)

def add6():
    import numpy as np
    from datasets import load_dataset
    dataset = load_dataset("mnist", split="train")

    for x in dataset:
        print(np.array(x['image']))
        print(x['label'])
        break


def get_meta_dataframe(meta_csv_path):

    if not os.path.isfile(meta_csv_path):

        os.makedirs(os.path.dirname(meta_csv_path), exist_ok=True)

        os.system("wget -O {} {}".format(meta_csv_path, "https://sandbox.zenodo.org/record/1186898/files/class_labels_indices.csv?download=1"))

        print("Download to {}".format(meta_csv_path))

    df = pd.read_csv(meta_csv_file, sep=',')

    return df


def add7():
    import os
    from pathlib import Path

    meta_csv_path = os.path.join(Path.home(), ".cache/metadata/class_labels_indices.csv")

    df = get_meta_dataframe(meta_csv_path)

    from IPython import embed; embed(using=False); os._exit(0)


def get_path_and_download(meta, re_download=False):

    path = meta["path"]
    remote_path = meta["remote_path"]
    size = meta["size"]

    if not Path(path).is_file() or Path(path).stat().st_size != size or re_download:

        Path(path).parents[0].mkdir(parents=True, exist_ok=True)
        os.system("wget -O {} {}".format(path, remote_path))
        print("Download to {}".format(path))

    return path


paths_dict = {
    "class_labels_indices.csv": {
        "path": Path(Path.home(), ".cache/metadata/class_labels_indices.csv"),
        "remote_path": "https://sandbox.zenodo.org/record/1186898/files/class_labels_indices.csv?download=1",
        "size": 14675,
    },
    "ontology.csv": {
        "path": Path(Path.home(), ".cache/metadata/ontology.json"),
        "remote_path": "https://sandbox.zenodo.org/record/1186898/files/ontology.json?download=1",
        "size": 342780,
    },
}


def add8():
    get_path(meta=paths_dict["ontology.csv"])


def add9():
    
    import museval
    import librosa
    import numpy as np

    sample_rate = 32000
    source_type = "vocals"

    target_path = "resources/queries/{}/{}.wav".format(source_type, source_type)
    # output_path = "separated_results/mixture/query={}.wav".format(source_type)
    output_path = "separated_results/mixture/query={}.wav".format(111)

    target, fs = librosa.load(path=target_path, sr=sample_rate, mono=True)
    output, fs = librosa.load(path=output_path, sr=sample_rate, mono=True)

    (sdrs, _, _, _) = museval.evaluate(target[None, :, None], output[None, :, None])  # (nsrc, nsampl, nchan)

    print(np.nanmedian(sdrs))
    
    from IPython import embed; embed(using=False); os._exit(0)


if __name__ == '__main__':

    add9()