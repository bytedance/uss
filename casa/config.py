import os
from pathlib import Path
import pandas as pd


def get_meta_csv_path(re_download=False):

    meta_csv_path = Path(Path.home(), ".cache/metadata/class_labels_indices.csv")

    if not meta_csv_path.is_file() or re_download:

        Path(meta_csv_path).parents[0].mkdir(parents=True, exist_ok=True)

        os.system("wget -O {} {}".format(meta_csv_path, "https://sandbox.zenodo.org/record/1186898/files/class_labels_indices.csv?download=1"))

        print("Download to {}".format(meta_csv_path))

    return meta_csv_path


def get_ontology_path(re_download=False):

    ontology_path = Path(Path.home(), ".cache/metadata/ontology.json")

    if not ontology_path.is_file() or re_download:

        Path(ontology_path).parents[0].mkdir(parents=True, exist_ok=True)

        os.system("wget -O {} {}".format(ontology_path, "https://sandbox.zenodo.org/record/1186898/files/ontology.json?download=1"))

        print("Download to {}".format(ontology_path))

    return ontology_path


SAMPLE_RATE = 32000
CLIP_SECONDS = 10.
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
FRAMES_PER_SECOND = 100

# Parse metadata
meta_csv_path = get_meta_csv_path()
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
