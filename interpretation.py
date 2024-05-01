import os
import pickle
from pathlib import Path

from config import DATA_DIR
from feature_extraction.AuDra_extended_model import load_feature_audra, load_original_audra
from feature_extraction.feature_extractor import process_images
from feature_extraction.feature_visualizer import create_dash_visualizer

#  USER EDIT (note: add your images to the 'user_images' folder)
output_filepath = f"{DATA_DIR}/Processed_images.pickle"

image_directory = Path(f"{DATA_DIR}/primary_images")


if not os.path.exists(output_filepath):
    print("Extracting features and principal components")
    feature_audra = load_feature_audra()
    original_model = load_original_audra()
    processed_data = process_images(original_model, feature_audra, image_directory)
    with open(output_filepath, "wb") as f:
        pickle.dump(processed_data, f)
else:
    print("Loading features and principal components")
    with open(output_filepath, "rb") as f:
        processed_data = pickle.load(f)


app = create_dash_visualizer(processed_data)

app.run_server(debug=True)