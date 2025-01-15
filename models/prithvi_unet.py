import math
import torch
import torch.nn as nn
from models.blocks import DownBlock, UpBlockWithSkip, Block, UpBlock
from models.random_half_dropout_layer import RandomHalfDropoutLayer
from models.prithvi_encoder import PrithviEncoder
    
class PrithviUNet(nn.Module):
    def __init__(self, in_channels, out_channels, weights_path, device, prithvi_encoder_size = 768, unet_encoder_size = 768, combine_method = 'concat', dropout_prob=2/3):
        super(PrithviUNet, self).__init__()
            
        assert (combine_method == 'concat') or (unet_encoder_size == prithvi_encoder_size), "The prithvi and unet encoder sizes must be the same"
        assert combine_method == 'concat' or dropout_prob == 0, "The dropout probability must be 0 when using add or mul combine methods"
        
        self.combine_method = combine_method
        
        def get_combine_fn(method):
            def combine(x1, x2):
                if method == 'concat':
                    return torch.cat([x1, x2], dim=1)
                elif method == 'add':
                    return x1 + x2
                elif method == 'mul':
                    return x1 * x2
                else:
                    raise ValueError(f"Unknown combine method: {method}")
            return combine
        
        self.combine = get_combine_fn(combine_method)

        # Encoder
        self.down1 = DownBlock(in_channels, in_channels*4) # 6 -> 24, 224 -> 112
        self.down2 = DownBlock(in_channels*4, in_channels*16) # 24 -> 96, 112 -> 56
        self.down3 = DownBlock(in_channels*16, in_channels*64) # 96 -> 384, 56 -> 28
        self.down4 = DownBlock(in_channels*64, unet_encoder_size) # 384 -> 768, 28 -> 14
        
        # Prithvi
        self.prithvi_encoder = PrithviEncoder(weights_path, device, target_channels=prithvi_encoder_size)
        
        # Training Switcher
        self.random_half_dropout = RandomHalfDropoutLayer(dropout_prob)
        
        # Decoder
        combine_size = prithvi_encoder_size + unet_encoder_size if combine_method == 'concat' else prithvi_encoder_size
        self.up1 = UpBlock(combine_size, in_channels*64) # 1536 -> 384, 14 -> 28
        self.up2 = UpBlockWithSkip(2*in_channels*64, in_channels*16) # 384 -> 96, 28 -> 56
        self.up3 = UpBlockWithSkip(2*in_channels*16, in_channels*4) # 96 -> 24, 56 -> 112
        self.up4 = UpBlockWithSkip(2*in_channels*4, in_channels) # 24 -> 6, 112 -> 224
        
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
    def change_prithvi_trainability(self, trainable):
        for param in self.prithvi_encoder.prithvi.parameters():
            param.requires_grad = trainable
        
    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        x_bottleneck = self.down4(x3)
        x_prithvi = self.prithvi_encoder(x)
        x = self.combine(x_bottleneck, x_prithvi)

        # mask on part to make training more robust
        x = self.random_half_dropout(x)
        
        # Decoder
        x = self.up1(x)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.out(x)
        return x