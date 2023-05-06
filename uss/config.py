from pathlib import Path

import pandas as pd

from uss.utils import get_path

csv_paths_dict = {
    "class_labels_indices.csv": {
        "path": Path(Path.home(), ".cache/uss/metadata/class_labels_indices.csv"),
        "remote_path": "https://sandbox.zenodo.org/record/1186898/files/class_labels_indices.csv?download=1",
        "size": 14675,
    },
    "ontology.csv": {
        "path": Path(Path.home(), ".cache/uss/metadata/ontology.json"),
        "remote_path": "https://sandbox.zenodo.org/record/1186898/files/ontology.json?download=1",
        "size": 342780,
    },
}

panns_paths_dict = {
    "Cnn14": {
        "path": Path(Path.home(), ".cache/panns/Cnn14_mAP=0.431.pth"),
        "remote_path": "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
        "size": 327428481,
    },
    "Cnn14_DecisionLevelMax": {
        "path": Path(Path.home(), ".cache/panns/Cnn14_DecisionLevelMax_mAP=0.385.pth"),
        "remote_path": "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1",
        "size": 327428481,
    },
}


SAMPLE_RATE = 32000
CLIP_SECONDS = 10.
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
FRAMES_PER_SECOND = 100

# Parse metadata
meta_csv_path = get_path(meta=csv_paths_dict["class_labels_indices.csv"])

df = pd.read_csv(meta_csv_path, sep=',')

IDS = df["mid"].tolist()
LABELS = df["display_name"].tolist()

CLASSES_NUM = len(LABELS)

LB_TO_IX = {label: i for i, label in enumerate(LABELS)}
IX_TO_LB = {i: label for i, label in enumerate(LABELS)}

ID_TO_IX = {id: i for i, id in enumerate(IDS)}
IX_TO_ID = {i: id for i, id in enumerate(IDS)}

ROOT_CLASS_ID_DICT = {
    "/m/0dgw9r": "Human sounds",
    "/m/0jbk": "Animal",
    "/m/04rlf": "Music",
    "/m/059j3w": "Natural sounds",
    "/t/dd00041": "Sounds of things",
    "/t/dd00098": "Source-ambiguous sounds",
    "/t/dd00123": "Channel, environment and background",
}
