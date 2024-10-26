import torch
from torch import nn
from models.prithvi_encoder import PrithviEncoder
from models.blocks import UpBlock
    
class PritviSegmenter(nn.Module):
    BOTTLENECK_WIN_SIZE = 14
    
    def __init__(self, weights_path, device):
        super(PritviSegmenter, self).__init__()
        self.device = device
        self.encoder = PrithviEncoder(weights_path, device)
        self.change_prithvi_trainability(False)
        
        # Decoder
        in_channels = self.BOTTLENECK_WIN_SIZE
        self.up1 = UpBlock(in_channels*128, in_channels*64) # 768 -> 384, 14 -> 28
        self.up2 = UpBlock(in_channels*64, in_channels*16) # 384 -> 96, 28 -> 56
        self.up3 = UpBlock(in_channels*16, in_channels*4) # 96 -> 24, 56 -> 112
        self.up4 = UpBlock(in_channels*4, in_channels) # 24 -> 6, 112 -> 224
        
    def change_prithvi_trainability(self, trainable):
        for param in self.encoder.parameters():
            param.requires_grad = trainable
        
    def forward(self, x):
        x = self.encoder(x)
        
        # Decoder
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        
        return x