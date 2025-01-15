import torch
from torch import nn
from models.prithvi_encoder import PrithviEncoder
from models.blocks import UpBlock, Block
    
class PritviSegmenter(nn.Module):
    BOTTLENECK_WIN_SIZE = 14
    
    def __init__(self, weights_path, device, output_channels, prithvi_encoder_size = 768):
        super(PritviSegmenter, self).__init__()
        self.device = device
        self.encoder = PrithviEncoder(weights_path, device, target_channels=prithvi_encoder_size).to(device)
        self.change_prithvi_trainability(False)
        self.output_channels = output_channels
        
        # Decoder
        self.up1 = UpBlock(prithvi_encoder_size, prithvi_encoder_size//2) # 768 -> 384, 14 -> 28
        self.up2 = UpBlock(prithvi_encoder_size//2, prithvi_encoder_size//8) # 384 -> 96, 28 -> 56
        self.up3 = UpBlock(prithvi_encoder_size//8, prithvi_encoder_size//32) # 96 -> 24, 56 -> 112
        self.up4 = UpBlock(prithvi_encoder_size//32, prithvi_encoder_size//64) # 24 -> 12, 112 -> 224
        
        self.final_block = Block(prithvi_encoder_size//64, self.output_channels)
        
    def change_prithvi_trainability(self, trainable):
        for param in self.encoder.prithvi.parameters():
            param.requires_grad = trainable
        
    def forward(self, x):
        x = self.encoder(x)
        
        # Decoder
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final_block(x)
        return x