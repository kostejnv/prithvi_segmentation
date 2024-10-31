from models.blocks import UpBlock
import torch
import yaml
from torch import nn
from prithvi.Prithvi import MaskedAutoencoderViT
from torchvision import transforms

class PrithviEncoder(nn.Module):
    OUTPUT_WINDOWS_SIZE = 14
    
    def __init__(self, weights_path, device, target_channels):
        super(PrithviEncoder, self).__init__()
        self.device = device
        self.model_args = None
        self.train_args = None
        self.encoder = None
        self.load_prithvi(weights_path, device)
        self.normalize = transforms.Normalize(self.train_args["data_mean"], self.train_args["data_std"])
        self.target_channels = target_channels
        self.final_conv = nn.Conv2d(self.model_args["embed_dim"], self.target_channels, kernel_size=3, padding=1, stride=1)
        
    def load_prithvi(self, weights_path, device):
        config_path = weights_path.replace('.pt', '_config.yaml')
        with open(config_path) as f:
            model_config = yaml.safe_load(f)
        self.model_args, self.train_args = model_config["model_args"], model_config["train_params"]
        self.model_args["num_frames"] = 1
        
        # create model
        model = MaskedAutoencoderViT(**self.model_args).to(self.device)
        model.eval()
        
        # load weights
        checkpoint = torch.load(weights_path, map_location=self.device)
        del checkpoint['pos_embed']
        del checkpoint['decoder_pos_embed']
        model.load_state_dict(checkpoint, strict=False)
        
        self.encoder = model.forward_encoder
        
    
    def forward(self, x):
        x = self.normalize(x)
        x = x.unsqueeze(2)
        features, _, _ = self.encoder(x, mask_ratio=0) # type: ignore
        reshaped_features = features[:, 1:, :]
        reshaped_features = reshaped_features.view(-1, self.OUTPUT_WINDOWS_SIZE, self.OUTPUT_WINDOWS_SIZE, self.model_args["embed_dim"])
        x= self.final_conv(reshaped_features.permute(0, 3, 1, 2))
        return x