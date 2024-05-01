from argparse import Namespace

import torch
from torch import nn

from AuDrA.AuDrA_Class import AuDrA
from config import DEVICE, ORIGINAL_AUDRA_SETTINGS


class FeatureAuDrA(nn.Module):
    def __init__(self, original_audra: AuDrA):
        super(FeatureAuDrA, self).__init__()
        self.features = nn.Sequential(
            *list(original_audra.model.children())[:-1]  # Remove the last fully connected block
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


def load_original_audra() -> AuDrA:
    model = AuDrA(Namespace(**ORIGINAL_AUDRA_SETTINGS))
    model_weights = torch.load('AuDrA/AuDrA_trained.ckpt', map_location=DEVICE)
    model.load_state_dict(model_weights['state_dict'])
    model.eval()
    model.to(DEVICE)
    return model


def load_feature_audra() -> FeatureAuDrA:
    original_audra: AuDrA = load_original_audra()
    truncated_model = FeatureAuDrA(original_audra)
    truncated_model.eval()
    truncated_model.to(DEVICE)
    return truncated_model
