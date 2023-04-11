import numpy as np
import pandas as pd
# import csv

SAMPLE_RATE = 32000
CLIP_SECONDS = 10.
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
FRAMES_PER_SECOND = 100

# Read csv file.
meta_csv_file = "./metadata/class_labels_indices.csv"
df = pd.read_csv(meta_csv_file, sep=',')

IDS = df['mid'].tolist()
LABELS = df['display_name'].tolist()

CLASSES_NUM = len(LABELS)

LB_TO_IX = {label : i for i, label in enumerate(LABELS)}
IX_TO_LB = {i : label for i, label in enumerate(LABELS)}

ID_TO_IX = {id : i for i, id in enumerate(IDS)}
IX_TO_ID = {i : id for i, id in enumerate(IDS)}
