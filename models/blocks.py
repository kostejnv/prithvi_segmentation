import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        # self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)  # Apply ReLU after the addition

        return out
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.resnet_block = ResNetBlock(in_channels)
        self.downscale = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.resnet_block(x)
        x = self.downscale(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class UpBlockWithSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlockWithSkip, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.resnet_block = ResNetBlock(in_channels)
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        x = self.resnet_block(x)
        x = self.up_conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.resnet_block = ResNetBlock(in_channels)
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.resnet_block(x)
        x = self.up_conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x