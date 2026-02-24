import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.wavelet_convolution import WaveConv1d, WaveConv2d

class WNO1d(nn.Module):
    def __init__(self, in_channels, out_channels, width, size, level=2, n_layers=4, padding=0):
        super(WNO1d, self).__init__()
        self.width = width
        self.size = size
        self.level = level
        self.n_layers = n_layers
        self.padding = padding
        self.in_channels = in_channels
        lifting_in = in_channels + 1 # +1 for x grid

        # Lifting Layer
        self.fc0 = nn.Linear(lifting_in, self.width)

        # Wavelet Layers
        padded_size = [s + padding for s in size] if isinstance(size, (list, tuple)) else [size + padding]
        self.conv_layers = nn.ModuleList([
            WaveConv1d(width, width, level, padded_size) 
            for _ in range(n_layers)
        ])

        # Linear projection layers (Skip connections)
        self.w_layers = nn.ModuleList([
            nn.Conv1d(width, width, 1)
            for _ in range(n_layers)
        ])

        # Projection Layers
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, 1, size_x).repeat([batchsize, 1, 1])
        return gridx

    def forward(self, x):
        # x input shape: (B, C, nx)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1) # (B, C+1, nx)
        
        # Lifting
        x = F.gelu(self.fc0(x.permute(0, 2, 1))).permute(0, 2, 1)
        
        if self.padding > 0:
            x = F.pad(x, [0, self.padding])
        
        for i in range(self.n_layers):
            x = self.conv_layers[i](x) + self.w_layers[i](x)
            if i < self.n_layers - 1:
                x = F.gelu(x)
                
        if self.padding > 0:
            x = x[..., :-self.padding]
            
        x = x.permute(0, 2, 1) # (B, nx, C)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 2, 1) # (B, out_channels, nx)
        return x

class WNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, width, size, level=2, n_layers=4, padding=2):
        super(WNO2d, self).__init__()
        self.width = width
        self.size = size
        self.level = level
        self.n_layers = n_layers
        self.padding = padding
        self.in_channels = in_channels
        lifting_in = in_channels + 2 # +2 for x,y grid

        # Lifting Layer
        self.fc0 = nn.Linear(lifting_in, self.width)

        # Wavelet Layers
        # Note: Size for WaveConv must account for padding
        padded_size = [s + padding for s in size]
        self.conv_layers = nn.ModuleList([
            WaveConv2d(width, width, level, padded_size) 
            for _ in range(n_layers)
        ])

        # Linear projection layers (Skip connections)
        self.w_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1)
            for _ in range(n_layers)
        ])

        # Projection Layers
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1)

    def forward(self, x):
        # x input shape: (B, C, nx, ny)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1) # (B, C+2, nx, ny)
        
        # Lifting using 1x1 conv to simulate Linear layer on per-point basis
        x = F.gelu(self.fc0(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        
        for i in range(self.n_layers):
            x = self.conv_layers[i](x) + self.w_layers[i](x)
            if i < self.n_layers - 1:
                x = F.gelu(x)
                
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
            
        x = x.permute(0, 2, 3, 1) # (B, nx, ny, C)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2) # (B, out_channels, nx, ny)
        return x