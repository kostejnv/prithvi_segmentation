import torch
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self, device):
        super(DiceLoss, self).__init__()
        self.device = device
        
    def forward(self, output, target):
        # Apply softmax to the output logits and use the class probabilities directly
        output = torch.softmax(output, dim=1)
        
        # Flatten the tensors
        output = output[:, 1].flatten()  # Assuming class 1 is the relevant one for binary dice
        target = target.flatten().float()
        
        # Ignore the '255' values in the target
        no_ignore = target.ne(255).to(self.device)
        output = output.masked_select(no_ignore)
        target = target.masked_select(no_ignore)
        
        # Compute the intersection and union for Dice calculation
        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target)
        dice = (2 * intersection + 1e-7) / (union + 1e-7)
        
        return 1 - dice