import matplotlib.pyplot as plt
import torchmetrics as TM
import torch
from torch import nn
import torchvision
import torchvision.transforms as T
import os
from models.ensemble_segmenter import EnsembleSegmenter

t2img = T.ToPILImage()
img2t = T.ToTensor()

def computeIOU(output, target, device):
  output = torch.argmax(output, dim=1).flatten() 
  target = target.flatten()
  
  no_ignore = target.ne(255).to(device)
  output = output.masked_select(no_ignore)
  target = target.masked_select(no_ignore)
  intersection = torch.sum(output * target)
  union = torch.sum(target) + torch.sum(output) - intersection
  iou = (intersection + .0000001) / (union + .0000001)
  
  if iou != iou:
    print("failed, replacing with 0")
    iou = torch.tensor(0).float()
  
  return iou

def computeMetrics(output, target, device):
  TP = truePositives(output, target, device).float()
  FN = falseNegatives(output, target, device).float()
  FP = falsePositives(output, target, device).float()
  TN = trueNegatives(output, target, device).float()
  
  return {
      "TP": TP,
      "FN": FN,
      "FP": FP,
      "TN": TN
  }
  
def computeAccuracy(output, target, device):
  output = torch.argmax(output, dim=1).flatten() 
  target = target.flatten()
  
  no_ignore = target.ne(255).to(device)
  output = output.masked_select(no_ignore)
  target = target.masked_select(no_ignore)
  correct = torch.sum(output.eq(target))
  
  return correct.float() / len(target)

def truePositives(output, target, device):
  output = torch.argmax(output, dim=1).flatten() 
  target = target.flatten()
  no_ignore = target.ne(255).to(device)
  output = output.masked_select(no_ignore)
  target = target.masked_select(no_ignore)
  correct = torch.sum(output * target)
  
  return correct

def trueNegatives(output, target, device):
  output = torch.argmax(output, dim=1).flatten() 
  target = target.flatten()
  no_ignore = target.ne(255).to(device)
  output = output.masked_select(no_ignore)
  target = target.masked_select(no_ignore)
  output = (output == 0)
  target = (target == 0)
  correct = torch.sum(output * target)
  
  return correct

def falsePositives(output, target, device):
  output = torch.argmax(output, dim=1).flatten() 
  target = target.flatten()
  no_ignore = target.ne(255).to(device)
  output = output.masked_select(no_ignore)
  target = target.masked_select(no_ignore)
  output = (output == 1)
  target = (target == 0)
  correct = torch.sum(output * target)
  
  return correct

def falseNegatives(output, target, device):
  output = torch.argmax(output, dim=1).flatten() 
  target = target.flatten()
  no_ignore = target.ne(255).to(device)
  output = output.masked_select(no_ignore)
  target = target.masked_select(no_ignore)
  output = (output == 0)
  target = (target == 1)
  correct = torch.sum(output * target)
  
  return correct

from enum import IntEnum
class TrimapClasses(IntEnum):
    PET = 0
    BACKGROUND = 1
    BORDER = 2

def close_figures():
    while len(plt.get_fignums()) > 0:
        plt.close()
    
def print_test_dataset_masks(model, inputs, labels, epoch, save_path, device):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs.to(device))
        
    if isinstance(model, EnsembleSegmenter):
        outputs = (outputs[0] + outputs[1]) / 2.0
        
    labels = labels.to(device).unsqueeze(1)
    predictions = torch.argmax(outputs, dim=1)
    predictions = predictions.unsqueeze(1)
    
    acc = computeAccuracy(outputs, labels, device)
    iou = computeIOU(outputs, labels, device)
    
    title = f'Epoch: {epoch:02d}, Accuracy: {acc:.4f}, IoU: {iou:.4f}'

    labels[labels == 255] = 2
    predictions[labels == 2] = 2
    # Close all previously open figures.
    close_figures()
    
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(title, fontsize=12)

    fig.add_subplot(3, 1, 1)
    print(inputs.shape)
    reorder = [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11]
    plt.imshow(t2img(torchvision.utils.make_grid(inputs[:, [2,1,0], :, :][reorder], nrow=6)))
    plt.axis('off')
    plt.title("Targets")

    fig.add_subplot(3, 1, 2)
    plt.imshow(t2img(torchvision.utils.make_grid(labels.float()[reorder] / 2.0, nrow=6)))
    plt.axis('off')
    plt.title("Ground Truth Labels")

    fig.add_subplot(3, 1, 3)
    plt.imshow(t2img(torchvision.utils.make_grid(predictions[reorder] / 2.0, nrow=6)))
    plt.axis('off')
    plt.title("Predicted Labels")
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"epoch_{epoch:02}.png"), format="png", bbox_inches="tight", pad_inches=0.4)
        
    close_figures()