import torch
from torch import nn

# Define a custom IoU Metric for validating the model.
def IoUMetric(pred, gt, softmax=False, num_classes=3):
    # Run softmax if input is logits.
    if softmax is True:
        pred = nn.Softmax(dim=1)(pred)
    
    # Add the one-hot encoded masks for all 3 output channels
    # (for all the classes) to a tensor named 'gt' (ground truth).
    gt = torch.cat([ (gt == i) for i in range(num_classes) ], dim=1)
    # print(f"[2] Pred shape: {pred.shape}, gt shape: {gt.shape}")

    intersection = gt * pred
    union = gt + pred - intersection

    # Compute the sum over all the dimensions except for the batch dimension.
    dim_tuple = tuple([i+1 for i in range(num_classes)])
    iou = (intersection.sum(dim=dim_tuple) + 0.001) / (union.sum(dim=dim_tuple) + 0.001)
    
    # Compute the mean over the batch dimension.
    return iou.mean()

class IoULoss(nn.Module):
    def __init__(self, softmax=False, num_classes=3):
        super().__init__()
        self.softmax = softmax
        self.num_classes = num_classes
    
    # pred => Predictions (logits, B, 3, H, W)
    # gt => Ground Truth Labales (B, 1, H, W)
    def forward(self, pred, gt):
        # return 1.0 - IoUMetric(pred, gt, self.softmax)
        # Compute the negative log loss for stable training.
        return -(IoUMetric(pred, gt, self.softmax, self.num_classes).log())