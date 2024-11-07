import torch
from torch import nn

# Define the LayerDropout class (assuming it's already imported)
class RandomHalfDropoutLayer(nn.Module):
    def __init__(self):
        super(RandomHalfDropoutLayer, self).__init__()

    def forward(self, x):
        if not self.training:
            return x  # No dropout in evaluation mode
        
        batch_size, num_channels, _, _ = x.size()
        half_channels = num_channels // 2
        
        # Create a random strategy tensor for the whole batch (0: mask upper half, 1: mask lower half, 2: no masking)
        strategies = torch.randint(0, 3, (batch_size, 1, 1, 1), device=x.device)
        
        # Create masks for upper and lower halves
        upper_mask = torch.ones_like(x)
        lower_mask = torch.ones_like(x)
        
        # Mask the upper half
        upper_mask[:, :half_channels, :, :] = 0
        # Mask the lower half
        lower_mask[:, half_channels:, :, :] = 0
        
        # Apply the strategies to create a final mask and scaling
        # Case 0: Apply upper mask and scale lower half by 2
        x = torch.where(strategies == 0, x * lower_mask * 2, x)
        
        # Case 1: Apply lower mask and scale upper half by 2
        x = torch.where(strategies == 1, x * upper_mask * 2, x)
        
        # Case 2: No masking (leave x as it is)
        return x


def test():
    # Initialize the LayerDropout layer
    layer_dropout = RandomHalfDropoutLayer()
    layer_dropout.train()  # Set to training mode to enable dropout

    # Create a sample input tensor (batch size 4, 6 channels, 3x3 spatial dimensions for simplicity)
    input_tensor = torch.randn(10, 6, 3, 3)

    # Apply the LayerDropout to the input
    output_tensor = layer_dropout(input_tensor)

    # Check and print results
    # print("Input Tensor:")
    # print(input_tensor)
    # print("\nOutput Tensor after LayerDropout:")
    # print(output_tensor)

    # Verifying the dropout effects for each sample in the batch
    for i in range(input_tensor.size(0)):
        print(f"\nSample {i+1}:")
        
        upper_half_input = input_tensor[i, :3, :, :].mean().item()  # Mean of upper half in the input
        lower_half_input = input_tensor[i, 3:, :, :].mean().item()  # Mean of lower half in the input
        
        upper_half_output = output_tensor[i, :3, :, :].mean().item()  # Mean of upper half in the output
        lower_half_output = output_tensor[i, 3:, :, :].mean().item()  # Mean of lower half in the output

        print(f"  Upper Half (Input): {upper_half_input:.2f}, Upper Half (Output): {upper_half_output:.2f}")
        print(f"  Lower Half (Input): {lower_half_input:.2f}, Lower Half (Output): {lower_half_output:.2f}")
        
        # Check if upper or lower half is zeroed and if the remaining half is scaled by ~2
        if upper_half_output == 0 and lower_half_output != 0:
            print("  Masked Upper Half, Lower Half Scaled by 2")
        elif lower_half_output == 0 and upper_half_output != 0:
            print("  Masked Lower Half, Upper Half Scaled by 2")
        else:
            print("  No Masking Applied")
            
            
if __name__ == "__main__":
    test()