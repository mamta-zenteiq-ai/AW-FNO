import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT, IDWT, DWT1D, IDWT1D

class WaveConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet='db4', mode='periodic'):
        super(WaveConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.wavelet = wavelet       
        self.mode = mode
        
        # Periodic mode matches FNO's assumption and often helps convergence
        self.dwt = DWT1D(J=self.level, mode=self.mode, wave=self.wavelet)
        self.idwt = IDWT1D(mode=self.mode, wave=self.wavelet)

        if isinstance(size, (list, tuple)):
            dummy_size = size
        else:
            dummy_size = (size,)
            
        dummy_data = torch.randn(1, 1, *dummy_size)        
        mode_data, mode_coefs = self.dwt(dummy_data)
        
        # Use a slightly larger scale to ensure the wavelet branch has an influence from the start
        self.scale = 0.05
        self.weight_approx = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, mode_data.shape[-1]))
        
        self.weight_details = nn.ParameterList([
            nn.Parameter(self.scale * torch.randn(in_channels, out_channels, c.shape[-1]))
            for c in mode_coefs
        ])

    def forward(self, x):
        x_ft, x_coeff = self.dwt(x)

        # Multiply the approximation (low-frequency)
        out_ft = torch.einsum("bix,iox->box", x_ft, self.weight_approx)
        
        # Multiply all detail levels
        out_coeff = []
        for i, c in enumerate(x_coeff):
            c_weighted = torch.einsum("bix,iox->box", c, self.weight_details[i])
            out_coeff.append(c_weighted)
            
        x = self.idwt((out_ft, out_coeff))
        return x

class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet='db4', mode='periodic'):
        super(WaveConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.wavelet = wavelet       
        self.mode = mode
        
        self.dwt = DWT(J=self.level, mode=self.mode, wave=self.wavelet)
        self.idwt = IDWT(mode=self.mode, wave=self.wavelet)

        if isinstance(size, (list, tuple)):
            dummy_size = size
        else:
            dummy_size = (size, size)
            
        dummy_data = torch.randn(1, 1, *dummy_size)        
        mode_data, mode_coefs = self.dwt(dummy_data)
        
        self.scale = 0.05
        self.weight_approx = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, mode_data.shape[-2], mode_data.shape[-1]))
        
        self.weight_details = nn.ParameterList()
        for c in mode_coefs:
            # (3 subbands, in, out, H, W)
            self.weight_details.append(nn.Parameter(self.scale * torch.randn(3, in_channels, out_channels, c.shape[-2], c.shape[-1])))

    def forward(self, x):
        x_ft, x_coeff = self.dwt(x)

        # 1. Multiply the approximation modes
        out_ft = torch.einsum("bixy,ioxy->boxy", x_ft, self.weight_approx)
        
        # 2. Process ALL detail coefficients
        out_coeff = []
        for i, c in enumerate(x_coeff):
            c_weighted = torch.einsum("bisxy,sioxy->bosxy", c, self.weight_details[i])
            out_coeff.append(c_weighted)
            
        x = self.idwt((out_ft, out_coeff))
        return x

class WaveConv3d(nn.Module):
    """
    3D Spatio-Temporal Wavelet Convolution using separable spatial and temporal 
    wavelet convolutions. This avoids requiring PTWT while building a purely 
    native PyTorch-Wavelets solution.
    """
    def __init__(self, in_channels, out_channels, level, size, wavelet='db4', mode='periodic'):
        super(WaveConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.size = size # (T, H, W)
        self.wavelet = wavelet
        self.mode = mode
        
        # Spatial Wavelet Convolution (H, W)
        self.wave_xy = WaveConv2d(in_channels, out_channels, level, (size[1], size[2]), wavelet=wavelet, mode=mode)
        # Temporal Wavelet Convolution (T)
        self.wave_t = WaveConv1d(in_channels, out_channels, level, size[0], wavelet=wavelet, mode=mode)
        
        # Feature mixing
        self.mixing = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)
        nn.init.constant_(self.mixing.weight, 0)
        nn.init.constant_(self.mixing.weight[:out_channels, :out_channels, :, :, :], 0.5)
        nn.init.constant_(self.mixing.weight[:out_channels, out_channels:, :, :, :], 0.5)
        nn.init.constant_(self.mixing.bias, 0)

    def forward(self, x):
        # x shape: (Batch, Channels, T, H, W)
        B, C, T, H, W = x.shape
        
        # 1. Processing spatial dimensions (H, W) for all times
        x_xy = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        out_xy = self.wave_xy(x_xy)
        # Reshape back to (B, C, T, H, W)
        out_xy = out_xy.reshape(B, T, self.out_channels, H, W).permute(0, 2, 1, 3, 4)
        
        # 2. Processing temporal dimension (T) for all spatial locations
        # Move T to the last dimension for 1D convolution: (B*H*W, C, T)
        x_t = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
        out_t = self.wave_t(x_t)
        # Slicing in case DWT/IDWT padded the temporal dimension (common for very short signals)
        out_t = out_t[..., :T]
        
        # Mix the spatial and temporal wave features
        expanded_out_t = out_t.reshape(B, H, W, self.out_channels, T).permute(0, 3, 4, 1, 2)
        return self.mixing(torch.cat([out_xy, expanded_out_t], dim=1))
