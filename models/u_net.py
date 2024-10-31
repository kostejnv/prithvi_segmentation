import math
import torch
import torch.nn as nn
from models.blocks import DownBlock, UpBlockWithSkip, ResNetBlock
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, unet_encoder_size=None):
        super(UNet, self).__init__()

        if unet_encoder_size is None:
            unet_encoder_size = in_channels * 128

        # Encoder
        self.down1 = DownBlock(in_channels, in_channels*4) # 6 -> 24, 224 -> 112
        self.down2 = DownBlock(in_channels*4, in_channels*16) # 24 -> 96, 112 -> 56
        self.down3 = DownBlock(in_channels*16, in_channels*64) # 96 -> 384, 56 -> 28
        self.down4 = DownBlock(in_channels*64, unet_encoder_size) # 384 -> 768, 28 -> 14
        
        # Bottleneck
        self.bottleneck = ResNetBlock(unet_encoder_size)
        
        # Decoder
        self.up1 = UpBlockWithSkip(unet_encoder_size, in_channels*64) # 768 -> 384, 14 -> 28
        self.up2 = UpBlockWithSkip(in_channels*64, in_channels*16) # 384 -> 96, 28 -> 56
        self.up3 = UpBlockWithSkip(in_channels*16, in_channels*4) # 96 -> 24, 56 -> 112
        self.up4 = UpBlockWithSkip(in_channels*4, in_channels) # 24 -> 6, 112 -> 224
        
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        # Bottleneck
        x = self.bottleneck(x4)
        
        # Decoder
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.out(x)
        return x