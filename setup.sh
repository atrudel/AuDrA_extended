if [ ! -d ".git" ]; then
    echo "Error: Please run this script from the root directory of the AuDrA_extended repository."
    exit 1
fi

echo "Creating conda environment"
conda env create -f environment.yml
conda activate audra

echo "Downloading the checkpoint of the trained AuDrA model"
curl -Lo AuDrA/AuDrA_trained.ckpt https://osf.io/download/eahmk/

echo "Downloading the dataset in a folder named 'Drawings'"
curl -Lo dataset.zip https://osf.io/download/9bu3a/
unzip dataset.zip
rm dataset.zip
mv "Audra Drawings" "Drawings"

echo "Setup complete"

