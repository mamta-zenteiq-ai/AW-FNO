import sys
import os
import torch

# Ensure the project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from awfno.models.awfno import AWFNO2d

def test():
    print("Testing AWFNO2d initialization and forward pass...")
    # Parameters for the model
    # in_channels: 3 (e.g., u, v, p or x, y coordinates + forcing)
    # out_channels: 1 (e.g., solution field)
    # n_modes: (16, 16) for Fourier branch
    # size: (64, 64) spatial resolution for Wavelet branch initialization
    # hidden_channels: 32 latent dimension
    model = AWFNO2d(
        in_channels=3,
        out_channels=1,
        n_modes=(16, 16),
        size=(64, 64),
        hidden_channels=32,
        n_layers=2,
        wno_level=2,
        padding=4
    )
    
    # Input tensor: (Batch, Channels, Height, Width)
    x = torch.randn(2, 3, 64, 64)
    
    # Forward pass
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    assert y.shape == (2, 1, 64, 64), f"Wrong output shape: {y.shape}"
    print("Success!")

if __name__ == "__main__":
    test()
