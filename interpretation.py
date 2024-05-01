import os
import pickle
from pathlib import Path

import dash
from dash import dcc
import plotly.graph_objs as go
from dash import html, no_update, callback
from dash.dependencies import Input, Output

from config import DATA_DIR
from feature_extraction.AuDra_extended_model import load_feature_audra, load_original_audra
from feature_extraction.feature_extractor import generate_features
from feature_extraction.feature_visualizer import create_dash_visualizer

#  USER EDIT (note: add your images to the 'user_images' folder)
output_filename = "AuDrA_tsne.pickle"
image_directory = Path(f"{DATA_DIR}/primary_images/Images_4")



if not os.path.exists(output_filename):
    print("Creating tsne features")
    feature_audra = load_feature_audra()
    original_audra = load_original_audra()
    processed_data = generate_features(original_audra, feature_audra, image_directory)
    with open(output_filename, "wb") as f:
        pickle.dump(processed_data, f)
else:
    print("Loading tsne features")
    with open(output_filename, "rb") as f:
        processed_data = pickle.load(f)


app = create_dash_visualizer(processed_data)

app.run_server(debug=True)