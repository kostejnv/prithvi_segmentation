import argparse
import logging
import sys
import torch
from torch import nn
from models.u_net import UNet
from models.prithvi_segmenter import PritviSegmenter
from models.prithvi_unet import PrithviUNet
import os
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from enum import Enum
from utils.testing import computeIOU, computeAccuracy, print_test_dataset_masks, computeMetrics
from data_loading.sen1floods11 import get_loader

torch.manual_seed(124)

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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a Segmentation model')
    parser.add_argument('--data_path', type=str, default='./data/sen1floods11', help='Path to the data directory.')
    parser.add_argument('--version', type=str, default='testing', help='Model version')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--model', type=str, default='prithvi_unet', help='Model to use for training (unet, prithvi_unet, prithvi)')
    
    parser.add_argument('--prithvi_out_channels', type=int, default=768, help='If set, force number of output channels from the Prithvi encoders')
    parser.add_argument('--unet_out_channels', type=int, default=768, help='If set, force number of output channels from the UNet encoders')
    parser.add_argument('--prithvi_finetune_ratio', type=float, default=1, help='Expects positive float. If set, Prithvi will be finetuned at 0.1 * learning_rate for the set number of additional epochs, with respect to original epoch count. (if set to 1.5 and epochs=100, train for additional 150 epochs)')
    parser.add_argument('--save_model_interval', type=int, default=5, help='Save the model every n epochs')
    parser.add_argument('--test_interval', type=int, default=5, help='Test the model every n epochs')
    return parser.parse_args()
    
# Get arguments
args = parse_arguments()
args.num_classes = 2
args.in_channels = 6

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('mps') if torch.backends.mps.is_available() else device
logger.info(f'Using device: {device}')

# Set up logging
args.version = f"{args.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
args.log_dir = f'./logs/{args.model}_{args.version}'
os.makedirs(args.log_dir, exist_ok=True)
args.model_dir = os.path.join(args.log_dir, 'models')
os.makedirs(args.model_dir, exist_ok=True)

writer = SummaryWriter(args.log_dir)

# Train the model for a single epoch
def train_model(model, loader, optimizer, criterion, epoch):
    model.train().to(device)

    running_loss = 0.0
    running_samples = 0
    running_accuracies = 0
    running_iou = 0
    
    for batch_idx, (imgs, masks) in enumerate(tqdm(loader), 0):
        optimizer.zero_grad()
        imgs = imgs.to(device)
        masks = masks.to(device)
        outputs = model(imgs)
        targets = masks.squeeze(1)

        loss = criterion(outputs, targets.long())
        iou = computeIOU(outputs, targets, device)
        accuracy = computeAccuracy(outputs, targets, device)
        
        loss.backward()
        optimizer.step()
    
        running_samples += targets.size(0)
        running_loss += loss.item()
        running_accuracies += accuracy
        running_iou += iou
        
    logger.info("Trained {} samples, Loss: {:.4f}, Accuracy: {:.4f}, IoU: {:.4f}".format(
        running_samples,
        running_loss / (batch_idx+1),
        running_accuracies / (batch_idx+1),
        running_iou / (batch_idx+1)
    ))
    writer.add_scalar("Loss/train", running_loss / (batch_idx+1), epoch)
    writer.add_scalar("Accuracy/train", running_accuracies / (batch_idx+1), epoch)
    writer.add_scalar("IoU/train", running_iou / (batch_idx+1), epoch)
    
def test(model, loader, epoch, set_name='valid'):
    model.eval()
    metricss = {}
    index = 0
    
    for (imgs, masks) in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        predictions = model(imgs)
        
        metrics = computeMetrics(predictions, masks, device)
        
        metricss = {k: metricss.get(k, 0) + v for k, v in metrics.items()}
        index += 1
        
        del imgs
        del masks
        del predictions
    
    TP, FP, TN, FN = metricss['TP'].item(), metricss['FP'].item(), metricss['TN'].item(), metricss['FN'].item()
    
    IOU_floods = TP / (TP + FN + FP)
    IOU_non_floods = TN / (TN + FP + FN)
    Avg_IOU = (IOU_floods + IOU_non_floods) / 2

    ACC_floods = TP / (TP + FN)
    ACC_non_floods = TN / (TN + FP)
    Avg_ACC = (ACC_floods + ACC_non_floods) / 2
    
    logger.info(f"{set_name}: IOU Floods: {IOU_floods:.4f}, IOU Non-Floods: {IOU_non_floods:.4f}, Avg IOU: {Avg_IOU:.4f}")
    logger.info(f"{set_name}: ACC Floods: {ACC_floods:.4f}, ACC Non-Floods: {ACC_non_floods:.4f}, Avg ACC: {Avg_ACC:.4f}")
    
    writer.add_scalar(f"IOU_floods/{set_name}", IOU_floods, epoch)
    writer.add_scalar(f"IOU_non_floods/{set_name}", IOU_non_floods, epoch)
    writer.add_scalar(f"Avg_IOU/{set_name}", Avg_IOU, epoch)
    writer.add_scalar(f"ACC_floods/{set_name}", ACC_floods, epoch)
    writer.add_scalar(f"ACC_non_floods/{set_name}", ACC_non_floods, epoch)
    writer.add_scalar(f"Avg_ACC/{set_name}", Avg_ACC, epoch)
    
def get_number_of_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
def main(args):
    train_loader = get_loader(args.data_path, DatasetType.TRAIN.value, args)
    valid_loader = get_loader(args.data_path, DatasetType.VALID.value, args)
    test_loader = get_loader(args.data_path, DatasetType.TEST.value, args)
    bolivia_loader = get_loader(args.data_path, DatasetType.BOLIVIA.value, args)
    
    match args.model:
        case 'unet':
            model = UNet(in_channels=args.in_channels, out_channels=args.num_classes, unet_encoder_size=args.unet_out_channels)
        case 'prithvi_unet':
            model = PrithviUNet(in_channels=args.in_channels, out_channels=args.num_classes, weights_path='./prithvi/Prithvi_100M.pt', device=device, prithvi_encoder_size=args.prithvi_out_channels, unet_encoder_size=args.unet_out_channels)
        case 'prithvi':
            model = PritviSegmenter(weights_path='./prithvi/Prithvi_100M.pt', device=device, output_channels=args.num_classes, prithvi_encoder_size=args.prithvi_out_channels)
    model = model.to(device)
    
    valid_samples = next(iter(valid_loader))
    
    def train_model_for_n_epochs(model, train_loader, optimizer, criterion, scheduler, epoch_range):
        for epoch in range(*epoch_range):
            logger.info(f"Epoch {epoch+1}/{epoch_range[1]}")
            train_model(model, train_loader, optimizer, criterion, epoch)
            scheduler.step()
            
            # Testing and saving
            if (epoch+1) % args.test_interval == 0:
                test(model, valid_loader, epoch)
                print_test_dataset_masks(model, valid_samples[0], valid_samples[1], epoch+1, args.log_dir, device)
            if (epoch+1) % args.save_model_interval == 0:
                torch.save(model.state_dict(), os.path.join(args.model_dir, f"model_{args.version}_{epoch+1}.pt"))
    # Set up the optimizer and loss function
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.3,0.7]).float().to(device), ignore_index=255)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, args.epochs)
    
    # Freeze Prithvi weights
    if 'prithvi' in args.model:
        model.change_prithvi_trainability(False)
    logger.info(f"Number of trainable parameters: {get_number_of_trainable_parameters(model)}")
    
                
    # Train the model
    train_model_for_n_epochs(model, train_loader, optimizer, criterion, scheduler, (0, args.epochs))
    
    if 'prithvi' in args.model and args.prithvi_finetune_ratio is not None:
            logger.info("Fine-tunning Prithvi")
            
            finetune_epochs = int(args.epochs * args.prithvi_finetune_ratio)
            model.change_prithvi_trainability(True)
            model.to(device)
            
            logger.info(f"Number of trainable parameters: {get_number_of_trainable_parameters(model)}")
            
            # Update the learning rate
            args.learning_rate = args.learning_rate * 0.1
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, finetune_epochs)
            
            train_model_for_n_epochs(model, train_loader, optimizer, criterion, scheduler, (args.epochs, args.epochs + finetune_epochs))

    # Test the model
    test(model, test_loader, args.epochs, 'test')
    test(model, bolivia_loader, args.epochs, 'bolivia')

if __name__ == '__main__':
    main(args)