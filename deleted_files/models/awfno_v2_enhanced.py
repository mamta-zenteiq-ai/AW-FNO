"""
AW-FNO v2 with Enhanced Gated Fusion.

Drop-in replacements for AWFNOv2_{1,2,3}d that accept a fusion_type parameter
selecting one of three improved fusion strategies:

  fusion_type='dual'  — DualGatedFusion: independent α_f, α_w gates,
                         no convex-combination constraint.
  fusion_type='se'    — SEGatedFusion: baseline local gate + SE channel-
                         attention bias from global average pool.
  fusion_type='cross' — CrossModalFusion: cross-branch modulation then
                         SE dual-gate fusion. (default, most expressive)

All other constructor arguments are identical to the original AWFNOv2 classes
so existing training scripts need only two changes:
  1. Import from this module instead of awfno_v2.
  2. Optionally pass fusion_type=... to the constructor.

Architecture (unchanged from v2 except the gate module):
  Input → Lifting → ┌ FourierBranch ─┐
                    │                ├─ EnhancedGate ─→ Projection
                    └ WaveletBranch ─┘
"""

import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from ..layers.spectral_convolution import SpectralConv
from ..layers.wavelet_convolution import WaveConv1d, WaveConv2d
from ..layers.embeddings import GridEmbedding2D, GridEmbeddingND
from ..layers.channel_mlp import ChannelMLP
from .base_model import BaseModel
from .awfno_v2 import Branch1d, Branch2d, Branch3d
from .enhanced_gated_fusion import (
    DualGatedFusion1d, SEGatedFusion1d, CrossModalFusion1d,
    DualGatedFusion2d, SEGatedFusion2d, CrossModalFusion2d,
    DualGatedFusion3d, SEGatedFusion3d, CrossModalFusion3d,
)

Number = Union[float, int]

_FUSION_MAP = {
    '1d': {'dual': DualGatedFusion1d, 'se': SEGatedFusion1d, 'cross': CrossModalFusion1d},
    '2d': {'dual': DualGatedFusion2d, 'se': SEGatedFusion2d, 'cross': CrossModalFusion2d},
    '3d': {'dual': DualGatedFusion3d, 'se': SEGatedFusion3d, 'cross': CrossModalFusion3d},
}

def _build_gate(dim: str, fusion_type: str, channels: int, se_reduction: int):
    cls = _FUSION_MAP[dim][fusion_type]
    # dual variant does not use se_reduction
    if fusion_type == 'dual':
        return cls(channels)
    return cls(channels, se_reduction=se_reduction)


class AWFNOv2Enhanced_1d(BaseModel):
    """AWFNOv2_1d with selectable enhanced gated fusion.

    Args:
        fusion_type: 'dual' | 'se' | 'cross'  (default 'cross')
        se_reduction: bottleneck ratio for SE channel attention  (default 4)
        All other args identical to AWFNOv2_1d.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes,
        size,
        hidden_channels: int,
        n_fno_layers: int = 4,
        n_wno_layers: int = 4,
        wno_level: int = 3,
        wno_wavelet: str = 'db6',
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: str = "grid",
        non_linearity=F.gelu,
        padding: int = 0,
        dropout: float = 0.0,
        fusion_type: str = 'cross',
        se_reduction: int = 4,
    ):
        super().__init__()
        self.padding = padding
        self.pos_embed = GridEmbeddingND(in_channels, dim=1) if positional_embedding == "grid" else None

        lifting_in = self.pos_embed.out_channels if self.pos_embed else in_channels
        self.lifting = ChannelMLP(
            lifting_in, hidden_channels, int(hidden_channels * lifting_channel_ratio),
            2, 1, non_linearity, dropout,
        )
        f_convs = nn.ModuleList([
            SpectralConv(hidden_channels, hidden_channels, n_modes)
            for _ in range(n_fno_layers)
        ])
        self.fourier_branch = Branch1d(f_convs, hidden_channels, n_fno_layers, non_linearity)

        psize = [s + padding for s in size]
        w_convs = nn.ModuleList([
            WaveConv1d(hidden_channels, hidden_channels, wno_level, psize, wavelet=wno_wavelet)
            for _ in range(n_wno_layers)
        ])
        self.wavelet_branch = Branch1d(w_convs, hidden_channels, n_wno_layers, non_linearity)

        self.gate = _build_gate('1d', fusion_type, hidden_channels, se_reduction)
        self.projection = ChannelMLP(
            hidden_channels, out_channels, int(hidden_channels * projection_channel_ratio),
            2, 1, non_linearity, dropout,
        )

    def forward(self, x):
        if self.pos_embed:
            x = self.pos_embed(x)
        x = self.lifting(x)
        res = x
        if self.padding > 0:
            x = F.pad(x, [0, self.padding])
        v_f = self.fourier_branch(x)
        v_w = self.wavelet_branch(x)
        x = self.gate(v_f, v_w)
        if self.padding > 0:
            x = x[..., :-self.padding]
        return self.projection(x + res)


class AWFNOv2Enhanced_2d(BaseModel):
    """AWFNOv2_2d with selectable enhanced gated fusion.

    Args:
        fusion_type: 'dual' | 'se' | 'cross'  (default 'cross')
        se_reduction: bottleneck ratio for SE channel attention  (default 4)
        All other args identical to AWFNOv2_2d.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes,
        size,
        hidden_channels: int,
        n_fno_layers: int = 4,
        n_wno_layers: int = 4,
        wno_level: int = 3,
        wno_wavelet: str = 'db6',
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: str = "grid",
        non_linearity=F.gelu,
        padding: int = 0,
        dropout: float = 0.0,
        fusion_type: str = 'cross',
        se_reduction: int = 4,
    ):
        super().__init__()
        self.padding = padding
        self.pos_embed = GridEmbedding2D(in_channels) if positional_embedding == "grid" else None
        lifting_in = self.pos_embed.out_channels if self.pos_embed else in_channels
        self.lifting = ChannelMLP(
            lifting_in, hidden_channels, int(hidden_channels * lifting_channel_ratio),
            2, 2, non_linearity, dropout,
        )
        f_convs = nn.ModuleList([
            SpectralConv(hidden_channels, hidden_channels, n_modes)
            for _ in range(n_fno_layers)
        ])
        self.fourier_branch = Branch2d(f_convs, hidden_channels, n_fno_layers, non_linearity)

        psize = [s + padding for s in size]
        w_convs = nn.ModuleList([
            WaveConv2d(hidden_channels, hidden_channels, wno_level, psize, wavelet=wno_wavelet)
            for _ in range(n_wno_layers)
        ])
        self.wavelet_branch = Branch2d(w_convs, hidden_channels, n_wno_layers, non_linearity)

        self.gate = _build_gate('2d', fusion_type, hidden_channels, se_reduction)
        self.projection = ChannelMLP(
            hidden_channels, out_channels, int(hidden_channels * projection_channel_ratio),
            2, 2, non_linearity, dropout,
        )

    def forward(self, x):
        if self.pos_embed:
            x = self.pos_embed(x, batched=True)
        x = self.lifting(x)
        res = x
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        v_f = self.fourier_branch(x)
        v_w = self.wavelet_branch(x)
        x = self.gate(v_f, v_w)
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
        return self.projection(x + res)


class AWFNOv2Enhanced_3d(BaseModel):
    """AWFNOv2_3d with selectable enhanced gated fusion.

    Args:
        fusion_type: 'dual' | 'se' | 'cross'  (default 'cross')
        se_reduction: bottleneck ratio for SE channel attention  (default 4)
        All other args identical to AWFNOv2_3d.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes,
        size,
        hidden_channels: int,
        n_fno_layers: int = 4,
        n_wno_layers: int = 4,
        wno_level: int = 3,
        wno_wavelet: str = 'db6',
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: str = "grid",
        non_linearity=F.gelu,
        padding: int = 0,
        dropout: float = 0.0,
        fusion_type: str = 'cross',
        se_reduction: int = 4,
    ):
        super().__init__()
        self.padding = padding
        self.pos_embed = GridEmbeddingND(in_channels, dim=3) if positional_embedding == "grid" else None
        lifting_in = self.pos_embed.out_channels if self.pos_embed else in_channels
        self.lifting = ChannelMLP(
            lifting_in, hidden_channels, int(hidden_channels * lifting_channel_ratio),
            2, 3, non_linearity, dropout,
        )
        f_convs = nn.ModuleList([
            SpectralConv(hidden_channels, hidden_channels, n_modes)
            for _ in range(n_fno_layers)
        ])
        self.fourier_branch = Branch3d(f_convs, hidden_channels, n_fno_layers, non_linearity)

        from ..layers.wavelet_convolution import WaveConv3d
        psize = [s + padding for s in size]
        w_convs = nn.ModuleList([
            WaveConv3d(hidden_channels, hidden_channels, wno_level, psize, wavelet=wno_wavelet)
            for _ in range(n_wno_layers)
        ])
        self.wavelet_branch = Branch3d(w_convs, hidden_channels, n_wno_layers, non_linearity)

        self.gate = _build_gate('3d', fusion_type, hidden_channels, se_reduction)
        self.projection = ChannelMLP(
            hidden_channels, out_channels, int(hidden_channels * projection_channel_ratio),
            2, 3, non_linearity, dropout,
        )

    def forward(self, x):
        if self.pos_embed:
            x = self.pos_embed(x)
        x = self.lifting(x)
        res = x
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])
        v_f = self.fourier_branch(x)
        v_w = self.wavelet_branch(x)
        x = self.gate(v_f, v_w)
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding, :-self.padding]
        return self.projection(x + res)
