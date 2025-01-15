import numpy as np
import torch
import torch.nn as nn

class EnsembleSegmenter(nn.Module):
    def __init__(self, prithvi_segmenter, unet_segmenter):
        super(EnsembleSegmenter, self).__init__()
        self.prithvi_segmenter = prithvi_segmenter
        self.unet_segmenter = unet_segmenter

    def forward(self, x):
        # Predict using Prithvi Segmenter
        prithvi_predictions = self.prithvi_segmenter(x)
        # Predict using U-Net Segmenter
        unet_predictions = self.unet_segmenter(x)
        return prithvi_predictions, unet_predictions
