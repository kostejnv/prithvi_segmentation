from models.blocks import Block
import torch
import yaml
from torch import nn
from models.masked_autoencoder import MaskedAutoencoderViT
from torchvision import transforms

class PrithviEncoder(nn.Module):
    OUTPUT_WINDOWS_SIZE = 14
    
    def __init__(self, weights_path, device, target_channels):
        super(PrithviEncoder, self).__init__()
        self.device = device
        self.target_channels = target_channels
        self.model_args = None
        self.train_args = None
        self.prithvi = self.load_prithvi(weights_path, device)
        self.block = Block(self.model_args["embed_dim"], target_channels)
        
    def load_prithvi(self, weights_path, device):
        config_path = weights_path.replace('.pt', '_config.yaml')
        with open(config_path) as f:
            model_config = yaml.safe_load(f)
        self.model_args, self.train_args = model_config["model_args"], model_config["train_params"]
        self.model_args["num_frames"] = 1 # we are only using 1 frame at a time
        
        # create model
        model = MaskedAutoencoderViT(**self.model_args)
        
        # load weights
        checkpoint = torch.load(weights_path, map_location=device)
        del checkpoint['pos_embed']
        del checkpoint['decoder_pos_embed']
        model.load_state_dict(checkpoint, strict=False)
        model = model.to(device)
        del model.decoder_blocks
        return model
        
    
    def forward(self, x):
        x = x.unsqueeze(2)
        features = self.prithvi.forward_encoder(x)
        x = features[:, 1:, :]
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], 768, 14, 14)
        x = self.block(x)
        return x