
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
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--model', type=str, default='prithvi_unet', help='Model to use for training (unet, prithvi_unet, prithvi)')
    
    parser.add_argument('--prithvi_out_channels', type=int, default=768, help='If set, force number of output channels from the Prithvi encoders')
    parser.add_argument('--unet_out_channels', type=int, default=768, help='If set, force number of output channels from the UNet encoders')
    parser.add_argument('--prithvi_finetune_ratio', type=float, default=None, help='Expects positive float. If set, Prithvi will be finetuned at 0.1 * learning_rate for the set number of additional epochs, with respect to original epoch count. (if set to 1.5 and epochs=100, train for additional 150 epochs)')
    return parser.parse_args()
    
args = parse_arguments()
args.num_classes = 2
args.in_channels = 6
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('mps') if torch.backends.mps.is_available() else device
logger.info(f'Using device: {device}')

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
    
def test(model, loader, epoch):
    model.eval().to(device)    
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
    # end for
    
    
    TP, FP, TN, FN = metricss['TP'].item(), metricss['FP'].item(), metricss['TN'].item(), metricss['FN'].item()
    
    IOU_floods = TP / (TP + FN + FP)
    IOU_non_floods = TN / (TN + FP + FN)
    Avg_IOU = (IOU_floods + IOU_non_floods) / 2

    ACC_floods = TP / (TP + FN)
    ACC_non_floods = TN / (TN + FP)
    Avg_ACC = (ACC_floods + ACC_non_floods) / 2
    
    logger.info(f"Test: IOU Floods: {IOU_floods:.4f}, IOU Non-Floods: {IOU_non_floods:.4f}, Avg IOU: {Avg_IOU:.4f}")
    logger.info(f"Test: ACC Floods: {ACC_floods:.4f}, ACC Non-Floods: {ACC_non_floods:.4f}, Avg ACC: {Avg_ACC:.4f}")
    
    writer.add_scalar("IOU_floods/test", IOU_floods, epoch)
    writer.add_scalar("IOU_non_floods/test", IOU_non_floods, epoch)
    writer.add_scalar("Avg_IOU/test", Avg_IOU, epoch)
    writer.add_scalar("ACC_floods/test", ACC_floods, epoch)
    writer.add_scalar("ACC_non_floods/test", ACC_non_floods, epoch)
    writer.add_scalar("Avg_ACC/test", Avg_ACC, epoch)
        
def main(args):
    train_loader = get_loader(args.data_path, DatasetType.TRAIN.value, args)
    valid_loader = get_loader(args.data_path, DatasetType.VALID.value, args)
    
    match args.model:
        case 'unet':
            model = UNet(in_channels=args.in_channels, out_channels=args.num_classes, unet_encoder_size=args.unet_out_channels)
        case 'prithvi_unet':
            model = PrithviUNet(in_channels=args.in_channels, out_channels=args.num_classes, weights_path='./prithvi/Prithvi_100M.pt', device=device, prithvi_encoder_size=args.prithvi_out_channels, unet_encoder_size=args.unet_out_channels)
        case 'prithvi':
            model = PritviSegmenter(weights_path='./prithvi/Prithvi_100M.pt', device=device, output_channels=args.num_classes, prithvi_encoder_size=args.prithvi_out_channels)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([3,7]).float().to(device), ignore_index=255)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, args.epochs)
    # criterion = DiceLoss(device)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Display the number of parameters
    print(f"Total trainable parameters: {count_parameters(model)}")
    
    # print(model)
    print(args.num_classes)
    
    test_samples = next(iter(valid_loader))
    print_test_dataset_masks(model, test_samples[0], test_samples[1], 0, args.log_dir, device)
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        train_model(model, train_loader, optimizer, criterion, epoch)
        test(model, valid_loader, epoch)
        scheduler.step()        
        print_test_dataset_masks(model, test_samples[0], test_samples[1], epoch+1, args.log_dir, device)
        torch.save(model.state_dict(), os.path.join(args.model_dir, f"model_{args.version}_{epoch+1}.pt"))
    
    if 'prithvi' in args.model and args.prithvi_finetune_ratio is not None:
            print('Switching Prithvi training on...')
            finetune_epochs = int(args.epochs * args.prithvi_finetune_ratio)

            
            # Turn on prithvi training
            model.change_prithvi_trainability(True)
            model.to(device)
            
            # Update the learning rate
            args.learning_rate = args.learning_rate * 0.1
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, finetune_epochs)

            # Continue Training
            # TODO: Find a way to not repeat this
            for epoch in range(args.epochs, args.epochs + finetune_epochs + 1):
                logger.info(f"Fine-tunning Epoch {epoch+1}/{finetune_epochs}")
                train_model(model, train_loader, optimizer, criterion, epoch)
                test(model, valid_loader, epoch)
                scheduler.step()        
                print_test_dataset_masks(model, test_samples[0], test_samples[1], epoch+1, args.log_dir, device)
                torch.save(model.state_dict(), os.path.join(args.model_dir, f"model_finetune_{args.version}_{epoch+1}.pt"))

    
if __name__ == '__main__':
    main(args)