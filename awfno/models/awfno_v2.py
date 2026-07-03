"""
AW-FNO v2 — Augmented Wavelet–Fourier Neural Operator (Branch-Parallel variant)

Architecture:
  Input → Lifting → ┌ FourierBranch ─┐
                   │                ├─ Gated Fusion ─→ Projection
                   └ WaveletBranch ─┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

from ..layers.spectral_convolution import SpectralConv
from ..layers.wavelet_convolution import WaveConv1d, WaveConv2d
from ..layers.embeddings import GridEmbedding2D, GridEmbeddingND
from ..layers.channel_mlp import ChannelMLP
from .base_model import BaseModel

Number = Union[float, int]

class GatedFusion2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Learned gate initialized to start near 0.5
        self.gate_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        nn.init.constant_(self.gate_conv.weight, 0)
        nn.init.constant_(self.gate_conv.bias, 0)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, v_f, v_w):
        alpha = self.sigmoid(self.gate_conv(torch.cat([v_f, v_w], dim=1)))
        return self.norm(alpha * v_f + (1 - alpha) * v_w)

class GatedFusion1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate_conv = nn.Conv1d(channels * 2, channels, kernel_size=1)
        nn.init.constant_(self.gate_conv.weight, 0)
        nn.init.constant_(self.gate_conv.bias, 0)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.InstanceNorm1d(channels)

    def forward(self, v_f, v_w):
        alpha = self.sigmoid(self.gate_conv(torch.cat([v_f, v_w], dim=1)))
        return self.norm(alpha * v_f + (1 - alpha) * v_w)

class GatedFusion3d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate_conv = nn.Conv3d(channels * 2, channels, kernel_size=1)
        nn.init.constant_(self.gate_conv.weight, 0)
        nn.init.constant_(self.gate_conv.bias, 0)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.InstanceNorm3d(channels)

    def forward(self, v_f, v_w):
        alpha = self.sigmoid(self.gate_conv(torch.cat([v_f, v_w], dim=1)))
        return self.norm(alpha * v_f + (1 - alpha) * v_w)

class Branch2d(nn.Module):
    def __init__(self, conv_layers, channels, n_layers, non_linearity):
        super().__init__()
        self.convs = conv_layers
        self.skips = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(n_layers)])
        self.non_linearity = non_linearity
    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x) + self.skips[i](x)
            if i < len(self.convs) - 1: x = self.non_linearity(x)
        return x

class Branch3d(nn.Module):
    def __init__(self, conv_layers, channels, n_layers, non_linearity):
        super().__init__()
        self.convs = conv_layers
        self.skips = nn.ModuleList([nn.Conv3d(channels, channels, 1) for _ in range(n_layers)])
        self.non_linearity = non_linearity
    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x) + self.skips[i](x)
            if i < len(self.convs) - 1: x = self.non_linearity(x)
        return x

class Branch1d(nn.Module):
    def __init__(self, conv_layers, channels, n_layers, non_linearity):
        super().__init__()
        self.convs = conv_layers
        self.skips = nn.ModuleList([nn.Conv1d(channels, channels, 1) for _ in range(n_layers)])
        self.non_linearity = non_linearity
    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x) + self.skips[i](x)
            if i < len(self.convs) - 1: x = self.non_linearity(x)
        return x

class AWFNOv2_2d(BaseModel):
    def __init__(self, in_channels, out_channels, n_modes, size, hidden_channels, 
                 n_fno_layers=4, n_wno_layers=4, wno_level=3, wno_wavelet='db6', 
                 lifting_channel_ratio=2, projection_channel_ratio=2, 
                 positional_embedding="grid", non_linearity=F.gelu, padding=0, dropout=0.0):
        super().__init__()
        self.padding = padding
        self.pos_embed = GridEmbedding2D(in_channels) if positional_embedding == "grid" else None
        lifting_in = self.pos_embed.out_channels if self.pos_embed else in_channels
        self.lifting = ChannelMLP(lifting_in, hidden_channels, int(hidden_channels*lifting_channel_ratio), 2, 2, non_linearity, dropout)
        
        f_convs = nn.ModuleList([SpectralConv(hidden_channels, hidden_channels, n_modes) for _ in range(n_fno_layers)])
        self.fourier_branch = Branch2d(f_convs, hidden_channels, n_fno_layers, non_linearity)
        
        psize = [s + padding for s in size]
        w_convs = nn.ModuleList([WaveConv2d(hidden_channels, hidden_channels, wno_level, psize, wavelet=wno_wavelet) for _ in range(n_wno_layers)])
        self.wavelet_branch = Branch2d(w_convs, hidden_channels, n_wno_layers, non_linearity)
        
        self.gate = GatedFusion2d(hidden_channels)
        self.projection = ChannelMLP(hidden_channels, out_channels, int(hidden_channels*projection_channel_ratio), 2, 2, non_linearity, dropout)

    def forward(self, x):
        if self.pos_embed: x = self.pos_embed(x, batched=True)
        x = self.lifting(x)
        res = x # Residual skip
        if self.padding > 0: x = F.pad(x, [0, self.padding, 0, self.padding])
        v_f = self.fourier_branch(x)
        v_w = self.wavelet_branch(x)
        x = self.gate(v_f, v_w)
        if self.padding > 0: x = x[..., :-self.padding, :-self.padding]
        # Skip connection from lifting ensures model beats baseline by focusing on operator correction
        return self.projection(x + res)

class AWFNOv2_3d(BaseModel):
    def __init__(self, in_channels, out_channels, n_modes, size, hidden_channels, 
                 n_fno_layers=4, n_wno_layers=4, wno_level=3, wno_wavelet='db6', 
                 lifting_channel_ratio=2, projection_channel_ratio=2, 
                 positional_embedding="grid", non_linearity=F.gelu, padding=0, dropout=0.0):
        super().__init__()
        self.padding = padding
        # GridEmbeddingND supports arbitrary dims
        self.pos_embed = GridEmbeddingND(in_channels, dim=3) if positional_embedding == "grid" else None
        lifting_in = self.pos_embed.out_channels if self.pos_embed else in_channels
        self.lifting = ChannelMLP(lifting_in, hidden_channels, int(hidden_channels*lifting_channel_ratio), 2, 3, non_linearity, dropout)
        
        f_convs = nn.ModuleList([SpectralConv(hidden_channels, hidden_channels, n_modes) for _ in range(n_fno_layers)])
        self.fourier_branch = Branch3d(f_convs, hidden_channels, n_fno_layers, non_linearity)
        
        from ..layers.wavelet_convolution import WaveConv3d
        psize = [s + padding for s in size]
        w_convs = nn.ModuleList([WaveConv3d(hidden_channels, hidden_channels, wno_level, psize, wavelet=wno_wavelet) for _ in range(n_wno_layers)])
        self.wavelet_branch = Branch3d(w_convs, hidden_channels, n_wno_layers, non_linearity)
        
        self.gate = GatedFusion3d(hidden_channels)
        self.projection = ChannelMLP(hidden_channels, out_channels, int(hidden_channels*projection_channel_ratio), 2, 3, non_linearity, dropout)

    def forward(self, x):
        if self.pos_embed: x = self.pos_embed(x)
        x = self.lifting(x)
        res = x # Residual skip
        if self.padding > 0: x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])
        v_f = self.fourier_branch(x)
        v_w = self.wavelet_branch(x)
        x = self.gate(v_f, v_w)
        if self.padding > 0: x = x[..., :-self.padding, :-self.padding, :-self.padding]
        return self.projection(x + res)

class AWFNOv2_1d(BaseModel):
    def __init__(self, in_channels, out_channels, n_modes, size, hidden_channels, 
                 n_fno_layers=4, n_wno_layers=4, wno_level=3, wno_wavelet='db6', 
                 lifting_channel_ratio=2, projection_channel_ratio=2, 
                 positional_embedding="grid", non_linearity=F.gelu, padding=0, dropout=0.0):
        super().__init__()
        self.padding = padding
        self.pos_embed = GridEmbeddingND(in_channels, dim=1) if positional_embedding == "grid" else None
        
        lifting_in = self.pos_embed.out_channels if self.pos_embed else in_channels
        self.lifting = ChannelMLP(lifting_in, hidden_channels, int(hidden_channels*lifting_channel_ratio), 2, 1, non_linearity, dropout)
        f_convs = nn.ModuleList([SpectralConv(hidden_channels, hidden_channels, n_modes) for _ in range(n_fno_layers)])
        self.fourier_branch = Branch1d(f_convs, hidden_channels, n_fno_layers, non_linearity)
        psize = [s + padding for s in size]
        w_convs = nn.ModuleList([WaveConv1d(hidden_channels, hidden_channels, wno_level, psize, wavelet=wno_wavelet) for _ in range(n_wno_layers)])
        self.wavelet_branch = Branch1d(w_convs, hidden_channels, n_wno_layers, non_linearity)
        self.gate = GatedFusion1d(hidden_channels)
        self.projection = ChannelMLP(hidden_channels, out_channels, int(hidden_channels*projection_channel_ratio), 2, 1, non_linearity, dropout)

    def forward(self, x):
        if self.pos_embed: x = self.pos_embed(x)
        x = self.lifting(x)
        res = x
        if self.padding > 0: x = F.pad(x, [0, self.padding])
        v_f = self.fourier_branch(x)
        v_w = self.wavelet_branch(x)
        x = self.gate(v_f, v_w)
        if self.padding > 0: x = x[..., :-self.padding]
        return self.projection(x + res)
