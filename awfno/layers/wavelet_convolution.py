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
