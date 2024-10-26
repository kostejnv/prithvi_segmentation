# Comparision Prithvi with Unet and their combinations

This repository experiments with different architectures that uses Prithvi fundation model and Unet architecture. The sen1flood11 dataset is used for this purpose.

## Instalation

- Clone the repository
- Install all packages in `requirements.txt`
- Download the Prithvi model

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M
# rename to a valid python module name
mv Prithvi-100M prithvi
touch prithvi/__init__.py
```

- Download the dataset (school data server)

## Run

`python train.py --version "experiemnt-name" --model "model-name" --learning_rate 0.0005 --epochs 100 --data_path './data/path/to/dataset'`

Try to use this run

- `python train.py --version "first_experiment" --model "unet" --learning_rate 0.0005 --epochs 100 --data_path './data/path/to/dataset'`
- `python train.py --version "first_experiment" --model "prithvi_unet" --learning_rate 0.0005 --epochs 100 --data_path './data/path/to/dataset'`
- `python train.py --version "first_experiment" --model "prithvi" --learning_rate 0.0005 --epochs 100 --data_path './data/path/to/dataset'`