# import argparse
# import os
# import time
# import pickle
# import pathlib
# from typing import Dict, List

# import librosa
# import lightning.pytorch as pl
# import matplotlib.pyplot as plt
# import numpy as np
# import soundfile
# import torch
# import torch.nn as nn

# from casa.config import ID_TO_IX, LB_TO_IX, IX_TO_LB

import pkg_resources

pkg_resources.resource_filename('casa', 'data/')

from IPython import embed; embed(using=False); os._exit(0)