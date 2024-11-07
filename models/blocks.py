import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.block = Block(in_channels, out_channels)
        self.downscale = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.block(x)
        x = self.downscale(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class UpBlockWithSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlockWithSkip, self).__init__()
        self.block = Block(in_channels, in_channels)
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        x = self.up_conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.block = Block(in_channels, in_channels)
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.block(x)
        x = self.up_conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x