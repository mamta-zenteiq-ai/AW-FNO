import torch
import torch.nn as nn
from pytorch_wavelets import DWT, IDWT, DWT1D, IDWT1D

class WaveConv3d(nn.Module):
    # Separable Spatio-Temporal Wavelet Convolution
    def __init__(self, in_channels, out_channels, level, size, wavelet='db4', mode='periodic'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.size = size # (T, H, W)
        self.mode = mode
        self.wavelet = wavelet
        
        self.dwt2 = DWT(J=level, mode=mode, wave=wavelet)
        self.idwt2 = IDWT(mode=mode, wave=wavelet)
        
        self.dwt1 = DWT1D(J=level, mode=mode, wave=wavelet)
        self.idwt1 = IDWT1D(mode=mode, wave=wavelet)
        
        dummy_h_w = torch.randn(1, 1, size[1], size[2])
        dwt2_ft, dwt2_c = self.dwt2(dummy_h_w)
        
        dummy_t = torch.randn(1, 1, size[0])
        dwt1_ft, dwt1_c = self.dwt1(dummy_t)
        
        modes_t = dwt1_ft.shape[-1]
        modes_y = dwt2_ft.shape[-2]
        modes_x = dwt2_ft.shape[-1]
        
        self.scale = 0.05
        self.weight_approx = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes_t, modes_y, modes_x))
        
        # Let's just parameterize the approximation for spatial AND temporal
        # The true DWT would have 7 high-frequency subbands per level. 
        # A cascade has 1d subbands + 2d subbands. We can simplify by just using the spatial details and temporal approx, 
        # OR just parameterizing the `x_ft_2d` and `x_ft_1d`.

        # Let's do something simpler:
        # 1. 1D Wavelet conv over Time
        # 2. 2D Wavelet conv over Space
        self.w_t = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.w_xy = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Wait, if I do that it's not a Wavelet Convolution in 3D.
        # How do I define a 3D Wavelet Convolution correctly without ptwt?
