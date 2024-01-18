# Soil Erosion Prediction in Python

## Contents

This project consists of two scripts:
1. `magic.py` creates a dataset of crops around the coordinate indicated in the name
   of satellite images coming from the Sentinel-2 source (Copernicus). The coordinates come from
   the LUCAS dataset, which pairs a C-factor with a x-y coordinate on the globe.
2. `train.py` trains a ConvNet-based model to solve a regression task consisting in
   predicting the C-factor from the crops (image crop not plan crop) extracted in step 1.

N.B.: C-factor accounts for how land cover, crops and crop management cause soil loss to vary from
those losses occurring in bare fallow areas.

## Install

Set up your Python environment as follows (ordering is important):
```bash
pip install --upgrade pip
conda install -c conda-forge gdal
pip install rasterio opencv-python tqdm numpy scikit-learn wandb tmuxp tabulate pyright ruff-lsp
pip install torch torchvision torchaudio
pip install fire
```

