import torch
import torch.nn as nn
from pytorch_wavelets import DWT, IDWT, DWT1D, IDWT1D

class WaveConv3d_Pseudo(nn.Module):
    # A pseudo-3D wavelet convolution using 2D spatial DWT and 1D temporal DWT
    def __init__(self, in_channels, out_channels, level, size, wavelet='db4', mode='periodic'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.size = size # (T, H, W)
        
        self.dwt2 = DWT(J=level, mode=mode, wave=wavelet)
        self.idwt2 = IDWT(mode=mode, wave=wavelet)
        
        self.dwt1 = DWT1D(J=level, mode=mode, wave=wavelet)
        self.idwt1 = IDWT1D(mode=mode, wave=wavelet)
        
        # We need parameter shapes. Let's trace a dummy tensor!
        dummy = torch.randn(1, 1, size[0], size[1], size[2]) # (B, C, T, H, W)
        B, C, T, H, W = dummy.shape
        
        # 1. 2D spatial DWT
        dummy_spatial = dummy.view(B*T, C, H, W)
        yl, yh = self.dwt2(dummy_spatial)
        # yl: (B*T, C, H', W') -> (B, T, C, H', W') -> (B, C, H', W', T)
        
    def forward(self, x):
        pass

