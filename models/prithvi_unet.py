import math
import torch
import torch.nn as nn
from models.blocks import DownBlock, UpBlockWithSkip, ResNetBlock
from models.prithvi_encoder import PrithviEncoder
    
class PrithviUNet(nn.Module):
    def __init__(self, in_channels, out_channels, weights_path, device):
        super(PrithviUNet, self).__init__()
        # Encoder
        self.down1 = DownBlock(in_channels, in_channels*4) # 6 -> 24, 224 -> 112
        self.down2 = DownBlock(in_channels*4, in_channels*16) # 24 -> 96, 112 -> 56
        self.down3 = DownBlock(in_channels*16, in_channels*64) # 96 -> 384, 56 -> 28
        self.down4 = DownBlock(in_channels*64, in_channels*128) # 384 -> 768, 28 -> 14
        
        # Prithvi
        self.prithvi_encoder = PrithviEncoder(weights_path, device)
        self.change_prithvi_trainability(False)
        
        # Bottleneck
        self.bottleneck = ResNetBlock(in_channels*128)
        
        # Decoder
        self.up1 = UpBlockWithSkip(2*in_channels*128, in_channels*64) # 1536 -> 384, 14 -> 28
        self.up2 = UpBlockWithSkip(in_channels*64, in_channels*16) # 384 -> 96, 28 -> 56
        self.up3 = UpBlockWithSkip(in_channels*16, in_channels*4) # 96 -> 24, 56 -> 112
        self.up4 = UpBlockWithSkip(in_channels*4, in_channels) # 24 -> 6, 112 -> 224
        
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
    def change_prithvi_trainability(self, trainable):
        for param in self.prithvi_encoder.parameters():
            param.requires_grad = trainable
        
    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        # Bottleneck
        x_bottleneck = self.bottleneck(x4)
        x_prithvi = self.prithvi_encoder(x)
        x = torch.cat([x_bottleneck, x_prithvi], dim=1)
        
        # Decoder
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.out(x)
        return x