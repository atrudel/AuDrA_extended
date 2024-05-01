import base64
from argparse import Namespace
from io import BytesIO
from pathlib import Path
from typing import List
from sklearn.decomposition import PCA

import torch
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

from AuDrA.AuDrA_DataModule import user_Dataloader
from config import ORIGINAL_AUDRA_SETTINGS, DEVICE, DATA_DIR


def process_images(original_model, feature_model, directory: str):
    #  LOAD IMAGES FOR PREDICTIONS
    dataloader = user_Dataloader(args=Namespace(**ORIGINAL_AUDRA_SETTINGS), data_dir=directory)

    #  GET AuDrA_extended PREDICTIONS
    features = []
    originalities = []
    images_base64 = []
    for img in tqdm(dataloader):
        filename = img[0][0]
        x = img[1]
        x = x.to(DEVICE)
        embedding = feature_model.forward(x)
        features.append(embedding.detach().cpu())

        originality = original_model.forward(x)
        originalities.append(originality.detach().cpu())

        image_base64 = load_and_convert_image(directory / filename)
        images_base64.append(image_base64)

    features = torch.cat(features, dim=0)
    originalities = torch.cat(originalities, dim=0)
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(features.numpy())
    # tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    # tsne_results = tsne.fit_transform(features.numpy())

    return {
        'originalities': originalities,
        'features': features,
        'principal_components': principal_components,
        'images_base64': images_base64,
    }

# Load images and convert to base64 strings for embedding
def load_and_convert_image(filepath):
    with Image.open(filepath) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")  # Convert to PNG for consistent format
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return "data:image/png;base64, " + img_str
