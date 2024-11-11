# U-Prithvi: Integrating a GeoAI Foundation Model with UNet for Flood Inundation Mapping

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

This repository contains all code and resources for experiments conducted in our paper, U-Prithvi: Integrating a GeoAI Foundation Model with UNet for Flood Inundation Mapping. Our research introduces U-Prithvi, a novel framework that combines a GeoAI foundation model with a UNet architecture to enhance flood inundation mapping.

## Requirements

- Python 3.12

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-repo/segmentation-floods.git
   cd segmentation-floods
   ```

2. **Install all required packages**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Prithvi model**

   ```bash
   # Ensure git-lfs is installed (https://git-lfs.com)
   git lfs install
   git clone https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M
   # Rename to a valid Python module name
   mv Prithvi-100M prithvi
   touch prithvi/__init__.py
   ```

4. **Download the *Sen1Floods11* dataset**

   Follow the download instructions [here](https://github.com/cloudtostreet/Sen1Floods11). The folder structure should be as follows:

   ```
   sen1floods11/
   ├── LabelHand/
   ├── S1Hand/
   ├── S2Hand/
   └── splits/
   ```

## Run

To train the model, execute the following command:

```bash
python train.py *params*
```

This will start the training process using the specified dataset and model directory.

### Parameters

- `--data_dir`: Path to the dataset directory.
- `--model_dir`: Path to the Prithvi model directory.
- `--batch_size`: Batch size for training (default: 32).
- `--epochs`: Number of epochs for training (default: 50).
- `--learning_rate`: Learning rate for the optimizer (default: 0.001).
- `--checkpoint_dir`: Directory to save model checkpoints.
- `--log_dir`: Directory to save training logs.

