import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Optional

from ..layers.spectral_convolution import SpectralConv
from ..layers.wavelet_convolution import WaveConv1d, WaveConv2d
from ..layers.embeddings import GridEmbedding2D, GridEmbeddingND
from ..layers.channel_mlp import ChannelMLP
from ..layers.skip_connections import skip_connection
from .base_model import BaseModel

Number = Union[float, int]

class AdaptiveGatedFusion1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Changed from 1 to channels for per-channel gating
        GateConv = nn.Conv1d(channels * 2, channels, kernel_size=1)
        # Initialize weights to be very small to start with alpha around 0.5
        nn.init.constant_(GateConv.weight, 0)
        nn.init.constant_(GateConv.bias, 0)
        
        self.gate = nn.Sequential(
            GateConv,
            nn.Sigmoid()
        )
        
    def forward(self, v_fno, v_wno):
        cat_v = torch.cat([v_fno, v_wno], dim=1)
        alpha = self.gate(cat_v)
        return alpha * v_fno + (1 - alpha) * v_wno

class AWFNOBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        wno_size: Tuple[int],
        wno_level: int = 3,
        wno_wavelet: str = 'db6',
        non_linearity: nn.Module = F.gelu,
        dropout: float = 0.0,
        norm: Optional[str] = "layer_norm",
    ):
        super().__init__()
        self.fno_conv = SpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            bias=False
        )
        self.wno_conv = WaveConv1d(
            in_channels,
            out_channels,
            wno_level,
            wno_size,
            wavelet=wno_wavelet
        )
        self.skip = skip_connection(
            in_features=in_channels,
            out_features=out_channels,
            skip_type="linear",
            n_dim=1,
            bias=True
        )
        self.gfm = AdaptiveGatedFusion1d(out_channels)
        if norm == "layer_norm" or norm == "layer":
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()
        self.non_linearity = non_linearity
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        v_fno = self.fno_conv(x)
        v_wno = self.wno_conv(x)
        v_gated = self.gfm(v_fno, v_wno)
        x_skip = self.skip(x)
        
        # Apply normalization to the fused features
        out = v_gated + x_skip
        # Permute for LayerNorm (B, C, L) -> (B, L, C)
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        out = out.permute(0, 2, 1)
        
        out = self.non_linearity(out)
        return self.dropout(out)

class AWFNO1d(BaseModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        size: Tuple[int],
        hidden_channels: int,
        n_layers: int = 4,
        wno_level: int = 3,
        wno_wavelet: str = 'db6',
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: str = "grid",
        non_linearity: nn.Module = F.gelu,
        padding: int = 0,
        dropout: float = 0.0,
        norm: Optional[str] = "layer_norm",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.n_modes = n_modes
        self.size = size
        
        if positional_embedding == "grid":
            self.pos_embed = GridEmbeddingND(in_channels, dim=1)
            lifting_in = self.pos_embed.out_channels
        else:
            self.pos_embed = None
            lifting_in = in_channels

        lifting_channels = int(hidden_channels * lifting_channel_ratio)
        self.lifting = ChannelMLP(
            in_channels=lifting_in,
            out_channels=hidden_channels,
            hidden_channels=lifting_channels,
            n_layers=2,
            n_dim=1,
            non_linearity=non_linearity,
            dropout=dropout
        )

        padded_size = [s + padding for s in size]
        
        blocks = []
        for i in range(n_layers):
            blocks.append(AWFNOBlock1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=n_modes,
                wno_size=padded_size,
                wno_level=wno_level,
                wno_wavelet=wno_wavelet,
                non_linearity=non_linearity,
                dropout=dropout,
                norm=norm
            ))
        self.blocks = nn.ModuleList(blocks)

        projection_channels = int(hidden_channels * projection_channel_ratio)
        self.projection = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=1,
            non_linearity=non_linearity,
            dropout=dropout
        )
        
    def forward(self, x):
        if self.pos_embed is not None:
            x = self.pos_embed(x)
            
        x = self.lifting(x)
        
        if self.padding > 0:
            x = F.pad(x, [0, self.padding])
            
        for block in self.blocks:
            x = block(x)
            
        if self.padding > 0:
            x = x[..., :-self.padding]
            
        x = self.projection(x)
        return x

class AdaptiveGatedFusion2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Changed from 1 to channels for per-channel gating
        GateConv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        # Initialize weights to be very small to start with alpha around 0.5
        nn.init.constant_(GateConv.weight, 0)
        nn.init.constant_(GateConv.bias, 0)
        
        self.gate = nn.Sequential(
            GateConv,
            nn.Sigmoid()
        )
        
    def forward(self, v_fno, v_wno):
        # Concatenate along channel dimension
        cat_v = torch.cat([v_fno, v_wno], dim=1)
        # Per-channel gate map
        alpha = self.gate(cat_v)
        return alpha * v_fno + (1 - alpha) * v_wno

class AWFNOBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, int],
        wno_size: Tuple[int, int],
        wno_level: int = 3,
        wno_wavelet: str = 'db6',
        non_linearity: nn.Module = F.gelu,
        dropout: float = 0.0,
        norm: Optional[str] = "layer_norm",
    ):
        super().__init__()
        self.fno_conv = SpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            bias=False
        )
        self.wno_conv = WaveConv2d(
            in_channels,
            out_channels,
            wno_level,
            wno_size,
            wavelet=wno_wavelet
        )
        self.skip = skip_connection(
            in_features=in_channels,
            out_features=out_channels,
            skip_type="linear",
            n_dim=2,
            bias=True
        )
        # Gated Fusion Mechanism
        self.gfm = AdaptiveGatedFusion2d(out_channels)
        if norm == "layer_norm" or norm == "layer":
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()
        self.non_linearity = non_linearity
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        v_fno = self.fno_conv(x)
        v_wno = self.wno_conv(x)
        
        # 1. Combine using gated mechanism
        v_gated = self.gfm(v_fno, v_wno)
        
        # 2. Add skip connection
        x_skip = self.skip(x)
        out = v_gated + x_skip
        
        # Apply normalization (B, C, H, W) -> (B, H, W, C)
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)
        
        out = self.non_linearity(out)
        return self.dropout(out)

class AWFNO2d(BaseModel):
    """
    Augmented Wavelet Fourier Neural Operator (AW-FNO) model.
    Combines FNO and WNO using an Adaptive Gated Fusion Mechanism (GFM).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, int],
        size: Tuple[int, int],
        hidden_channels: int,
        n_layers: int = 4,
        wno_level: int = 3,
        wno_wavelet: str = 'db6',
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: str = "grid",
        non_linearity: nn.Module = F.gelu,
        padding: int = 0,
        dropout: float = 0.0,
        norm: Optional[str] = "layer_norm",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.n_modes = n_modes
        self.size = size
        
        # 1. Coordinate Encoding
        if positional_embedding == "grid":
            self.pos_embed = GridEmbedding2D(in_channels)
            lifting_in = self.pos_embed.out_channels
        else:
            self.pos_embed = None
            lifting_in = in_channels

        # 2. Lifting
        lifting_channels = int(hidden_channels * lifting_channel_ratio)
        self.lifting = ChannelMLP(
            in_channels=lifting_in,
            out_channels=hidden_channels,
            hidden_channels=lifting_channels,
            n_layers=2,
            n_dim=2,
            non_linearity=non_linearity,
            dropout=dropout
        )

        padded_size = [s + padding for s in size]
        
        # 3. AWFNO Blocks
        blocks = []
        for i in range(n_layers):
            blocks.append(AWFNOBlock2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=n_modes,
                wno_size=padded_size,
                wno_level=wno_level,
                wno_wavelet=wno_wavelet,
                non_linearity=non_linearity,
                dropout=dropout,
                norm=norm
            ))
        self.blocks = nn.ModuleList(blocks)

        # 4. Projection
        projection_channels = int(hidden_channels * projection_channel_ratio)
        self.projection = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=2,
            non_linearity=non_linearity,
            dropout=dropout
        )
        
    def forward(self, x):
        if self.pos_embed is not None:
            x = self.pos_embed(x, batched=True)
            
        x = self.lifting(x)
        
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
            
        for block in self.blocks:
            x = block(x)
            
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
            
        x = self.projection(x)
        return x
