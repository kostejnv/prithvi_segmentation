import os
import sys
import argparse
import logging
from enum import Enum
import torch
from data_loading.sen1floods11 import processTestIm
import rasterio
from models.u_net import UNet
from models.prithvi_segmenter import PritviSegmenter
from models.prithvi_unet import PrithviUNet
from PIL import Image  # Add this import
import numpy as np
import pandas as pd


class DatasetType(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    BOLIVIA = 'bolivia'

# Set up logging before importing TensorFlow
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

NUM_CLASSES = 2
IN_CHANNELS = 6
models_paths = {
    'unet': 'logs/unet_model_v1.0_100_20241109_144640_100.pt',
    'prithvi': 'logs/prithvi_model_v1.0_100epochs_20241108_150250_100.pt',
    'prithvi_unet': 'logs/model_v1.0_100epochs_20241108_152219_100.pt',
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a Segmentation model')
    parser.add_argument('csv_path', type=str, help='Path to the csv file containing first column as image path and second column as mask path.')
    parser.add_argument('--data_path', type=str, default='./data/sen1floods11', help='Path to the data directory.')
    # parser.add_argument('model_path', type=str, help='Path to the trained model.')
    parser.add_argument('output_path', type=str, help='Path to save the segmented image.')
    parser.add_argument('--bands', type=list, default=[1,2,3,8,11,12], help='Bands indices that need to be used for segmentation. Recall that the model was trained on 6 bands (B, G, R, NIR, SWIR1, SWIR2).')
    parser.add_argument('--model_type', type=str, default='prithvi_unet', help='Model to use for segmentation (unet, prithvi_unet, prithvi)')
    parser.add_argument('--weights_path', type=str, default='./prithvi/Prithvi_100M.pt', help='Path to the weights file for Prithvi models.')
    return parser.parse_args()

# Get arguments
args = parse_arguments()

def segment_image_with_sliding_window(model, image, window_size, stride, device):
    model.eval()  # Set the model to evaluation mode
    _, _, h_img, w_img = image.shape  # Get the height and width of the image
    h_stride, w_stride = stride  # Unpack the stride values
    h_crop, w_crop = window_size  # Unpack the window size values

    # Calculate the number of horizontal and vertical grids
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    # Initialize tensors to store the predictions and count matrix
    preds = torch.zeros((1, NUM_CLASSES, h_img, w_img), device=device)
    count_mat = torch.zeros((1, 1, h_img, w_img), device=device)

    # Loop over the grids
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            # Calculate the coordinates of the current window
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            # Extract the current window from the image
            crop_img = image[:, :, y1:y2, x1:x2]
            with torch.no_grad():  # Disable gradient calculation
                crop_seg_logit = model(crop_img)  # Perform segmentation on the current window
                crop_seg = torch.softmax(crop_seg_logit, dim=1)  # Apply softmax to get the probabilities

            # Accumulate the predictions and update the count matrix
            preds[:, :, y1:y2, x1:x2] += crop_seg
            count_mat[:, :, y1:y2, x1:x2] += 1

    # Normalize the predictions by the count matrix
    preds = preds / count_mat
    preds = torch.argmax(preds, dim=1)  # Get the class index with the highest probability
    return preds

def load_model(model_type, model_path, device, weights_path):
    if model_type == 'unet':
        model = UNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES)
    elif model_type == 'prithvi_unet':
        model = PrithviUNet(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES, weights_path=weights_path, device=device)
    elif model_type == 'prithvi':
        model = PritviSegmenter(output_channels=NUM_CLASSES, weights_path=weights_path, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    return model

PERCENTILES = (0.1, 99.9)
NO_DATA_FLOAT = 255

def enhance_mask_for_visualization(mask, no_data_pixel, gt):
    gt = gt.squeeze()
    mask = mask.squeeze() * 255
    print(mask.shape)
    mask[no_data_pixel] = 128
    # have 3 channels for visualization
    
    
    mask_extended = np.stack([mask, mask, mask], axis=2)
    
    # pixels that were wrongly classified label as red
    mask_extended[((mask == 0) & (gt == 1)) | ((mask == 255) & (gt == 0))] = [255, 0, 0]
    return mask_extended.astype(np.uint8)

def get_2_models_visualization(good, bad, gt):
    good, bad = good.squeeze(), bad.squeeze()
    gt = gt.squeeze()
    
    output = np.zeros((good.shape[0], good.shape[1], 3), dtype=np.uint8)
    output[good == 1] = [255, 255, 255]
    output[bad == 1] = [255, 255, 255]
    output[gt == -1] = [128, 128, 128]
    # both models were wrong = blue
    output[((good == 0) & (bad == 0) & (gt == 1)) | ((good == 1) & (bad == 1) & (gt == 0))] = [100, 100, 255]
    # unet was correct, prithvi was wrong = red
    output[((good == 0) & (bad == 1) & (gt == 1)) | ((good == 1) & (bad == 0) & (gt == 0))] = [255, 100, 100]
    # prithvi was correct, unet was wrong = green
    output[((good == 1) & (bad == 0) & (gt == 1)) | ((good == 0) & (bad == 1) & (gt == 0))] = [0, 255, 0]
    return output
    

def enhance_input_for_visualization(image):
    image = image.cpu().numpy()
    image = image.squeeze()[[2, 1, 0], :, :].transpose((1, 2, 0))
    mins, maxs = np.percentile(image, PERCENTILES)
    # Normalize the image between 0 and 255
    image = (image - mins) / (maxs - mins) * 255
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)
    
def add_blue_gap(images, gap_size=10):
    """Concatenate images with a bright blue gap between each pair."""
    total_width = sum(img.shape[1] for img in images) + (len(images) - 1) * gap_size
    max_height = max(img.shape[0] for img in images)
    result = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255  # Create a white canvas
    result[:, :, 0:2] = 0  # Set the blue channel to 0

    current_x = 0
    for img in images:
        result[:img.shape[0], current_x:current_x + img.shape[1]] = img
        current_x += img.shape[1] + gap_size

    return result

def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('mps') if torch.backends.mps.is_available() else device
    logger.info(f'Using device: {device}')

    
    # Load the image
    df = pd.read_csv(args.csv_path)
    
    models = {}
    model_names = ['unet', 'prithvi_unet', 'prithvi']
    for model_type in model_names:
        args.model_type = model_type
        models[model_type] = load_model(args.model_type, models_paths[model_type], device, args.weights_path)
        
    for i in range(len(df)):
        filename = df.iloc[i, 0]
        mask = df.iloc[i, 1]
        image = rasterio.open(os.path.join(args.data_path, 'S2Hand', filename)).read()
        mask = rasterio.open(os.path.join(args.data_path, 'LabelHand', mask)).read()
        image = processTestIm(image, args.bands).to(device)
    
        # Define window size and stride
        window_size = (224, 224)
        stride = (128, 128)

        predictions = {}
        # Perform segmentation
        for model_type, model in models.items():
            segmented_image = segment_image_with_sliding_window(model, image, window_size, stride, device)
            predictions[model_type] = segmented_image.cpu().numpy()

        print(mask.max(), mask.min())
        # Save the segmented image as a PNG file
        input_to_stored = enhance_input_for_visualization(image)
        # mask_to_stored = enhance_mask_for_visualization(mask, no_data_pixels, mask)
        # predictions_to_stored = []
        # for pred in predictions:
        #     prediction_to_stored = enhance_mask_for_visualization(pred.cpu().numpy(), no_data_pixels, mask)
        #     predictions_to_stored.append(prediction_to_stored)
        uprithvi_unet = get_2_models_visualization(predictions['prithvi_unet'], predictions['unet'], mask)
        
        uprithvi_prithvi = get_2_models_visualization(predictions['prithvi_unet'], predictions['prithvi'], mask)
            
        # final_image = add_blue_gap([input_to_stored, mask_to_stored, *predictions_to_stored])
        final_image = add_blue_gap([input_to_stored, uprithvi_unet, uprithvi_prithvi])

        final_image = Image.fromarray(final_image)
        final_image.save(os.path.join(args.output_path, f'{filename}.png'))
    

if __name__ == '__main__':
    main()