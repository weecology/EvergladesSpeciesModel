# EvergladesSpeciesModel
A deepforest model for wading bird species prediction.

## Environment

```
conda create --name EvergladesSpeciesModel python=3.8
conda activate EvergladesSpeciesModel
pip install DeepForest
```

# Workflow
0. This repo assumes that data has already been downloaded and cleaned from Zooniverse. If not please see [here](https://github.com/weecology/EvergladesWadingBird/blob/main/README.md#download-and-clean-zooniverse-annotations). This repo also assumes you have created a comet dashboard. Edit the workspace name here if needed. The script will look for comet authentication api_key locally, see [non-interactive setup](https://www.comet.ml/docs/python-sdk/advanced/)

1. Create training and test data splits

```
python create_species_model.py
```

2. Train a DeepForest Model

```
python everglades_species.py
```

