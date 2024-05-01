import os
from pathlib import Path

from torch import nn

root_directory = Path(os.path.dirname(os.path.realpath(__file__)))


DATA_DIR = f"{root_directory}/Drawings"
DEVICE = "cpu"

ORIGINAL_AUDRA_SETTINGS = {
        'architecture': 'resnet18',
        'pretrained': True,
        'in_shape': [3, 224, 224],
        'img_means': 0.1612,
        'img_stds': 0.4075,
        'num_outputs': 1,
        'learning_rate': 0.00034664640432471026,
        'batch_size': 16,
        'train_pct': 0.7,
        'val_pct': 0.1,
        'loss_func': nn.MSELoss(),
        'num_workers': 1
    }