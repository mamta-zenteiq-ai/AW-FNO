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

# ---------------------------------------------------------------------------
# ARCHITECTURE — AW-FNO with Final AGFM  (matches LaTeX description exactly)
# ---------------------------------------------------------------------------
#
#  i)  Lifting: (x, a(x)) ──P──> v_0  ∈ R^{B×C×H×W}
#
#  ii) Two INDEPENDENT T-layer branches, both fed v_0:
#
#        FNO branch:  v_0 → FNOBlock → v_1^F → ... → FNOBlock → v_FNO
#        WNO branch:  v_0 → WNOBlock → v_1^W → ... → WNOBlock → v_WNO
#
#      Each FNOBlock:  v_{t+1} = σ( LayerNorm( K_F(v_t) + W·v_t ) )
#      Each WNOBlock:  v_{t+1} = σ( LayerNorm( K_W(v_t) + W·v_t ) )
#      where K_F = SpectralConv, K_W = WaveConv, W·v_t = skip (pointwise linear).
#
# iii) Adaptive Gated Fusion (AGFM) — applied ONCE after both branches finish:
#
#        v_cat  = [v_FNO, v_WNO]          (concat along C, shape B×2C×H×W)
#        α      = sigmoid( Conv(v_cat) )   (shape B×C_gated×H×W)
#        v_fused = α · v_FNO + (1-α) · v_WNO
#
#      C_gated choices:
#        C_gated = C  →  per-channel gating (one α per spatial location per channel)
#        C_gated = 1  →  spatial gating (one α per spatial location, broadcast over C)
#      The description notes C_gated=1 works well empirically; it is the default here.
#
#  iv) Projection: v_fused ──Q──> u(x)  ∈ R^{d_u}
#
# Key difference from awfno.py / awfno_parallel.py:
#   Those files fuse FNO and WNO outputs at EVERY block (T fusions total).
#   This file fuses them ONCE at the very end (1 fusion total), which is what
#   the paper description says.
# ---------------------------------------------------------------------------


# ===========================================================================
# Shared building blocks
# ===========================================================================

class FNOBlock1d(nn.Module):
    """One FNO update step: v_{t+1} = σ( LN( SpectralConv(v_t) + skip(v_t) ) )"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        non_linearity=F.gelu,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fno_conv = SpectralConv(in_channels, out_channels, n_modes, bias=False)
        self.skip = skip_connection(in_channels, out_channels, skip_type="linear", n_dim=1, bias=True)
        self.norm = nn.LayerNorm(out_channels)
        self.non_linearity = non_linearity
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        # fno_conv and skip both read x independently — run them in parallel.
        future_conv = torch.jit.fork(self.fno_conv, x)
        future_skip = torch.jit.fork(self.skip,     x)
        out = torch.jit.wait(future_conv) + torch.jit.wait(future_skip)
        out = out.permute(0, 2, 1)      # (B,C,L) → (B,L,C) for LayerNorm
        out = self.norm(out)
        out = out.permute(0, 2, 1)      # (B,L,C) → (B,C,L)
        return self.dropout(self.non_linearity(out))


class WNOBlock1d(nn.Module):
    """One WNO update step: v_{t+1} = σ( LN( WaveConv(v_t) + skip(v_t) ) )"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        wno_size: Tuple[int],
        wno_level: int = 3,
        wno_wavelet: str = 'db6',
        non_linearity=F.gelu,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.wno_conv = WaveConv1d(in_channels, out_channels, wno_level, wno_size, wavelet=wno_wavelet)
        self.skip = skip_connection(in_channels, out_channels, skip_type="linear", n_dim=1, bias=True)
        self.norm = nn.LayerNorm(out_channels)
        self.non_linearity = non_linearity
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        # wno_conv and skip both read x independently — run them in parallel.
        future_conv = torch.jit.fork(self.wno_conv, x)
        future_skip = torch.jit.fork(self.skip,     x)
        out = torch.jit.wait(future_conv) + torch.jit.wait(future_skip)
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        out = out.permute(0, 2, 1)
        return self.dropout(self.non_linearity(out))


class FNOBlock2d(nn.Module):
    """One FNO update step: v_{t+1} = σ( LN( SpectralConv(v_t) + skip(v_t) ) )"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, int],
        non_linearity=F.gelu,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fno_conv = SpectralConv(in_channels, out_channels, n_modes, bias=False)
        self.skip = skip_connection(in_channels, out_channels, skip_type="linear", n_dim=2, bias=True)
        self.norm = nn.LayerNorm(out_channels)
        self.non_linearity = non_linearity
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        # fno_conv and skip both read x independently — run them in parallel.
        future_conv = torch.jit.fork(self.fno_conv, x)
        future_skip = torch.jit.fork(self.skip,     x)
        out = torch.jit.wait(future_conv) + torch.jit.wait(future_skip)
        out = out.permute(0, 2, 3, 1)   # (B,C,H,W) → (B,H,W,C) for LayerNorm
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)   # (B,H,W,C) → (B,C,H,W)
        return self.dropout(self.non_linearity(out))


class WNOBlock2d(nn.Module):
    """One WNO update step: v_{t+1} = σ( LN( WaveConv(v_t) + skip(v_t) ) )"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        wno_size: Tuple[int, int],
        wno_level: int = 3,
        wno_wavelet: str = 'db6',
        non_linearity=F.gelu,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.wno_conv = WaveConv2d(in_channels, out_channels, wno_level, wno_size, wavelet=wno_wavelet)
        self.skip = skip_connection(in_channels, out_channels, skip_type="linear", n_dim=2, bias=True)
        self.norm = nn.LayerNorm(out_channels)
        self.non_linearity = non_linearity
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        # wno_conv and skip both read x independently — run them in parallel.
        future_conv = torch.jit.fork(self.wno_conv, x)
        future_skip = torch.jit.fork(self.skip,     x)
        out = torch.jit.wait(future_conv) + torch.jit.wait(future_skip)
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)
        return self.dropout(self.non_linearity(out))


# ===========================================================================
# Adaptive Gated Fusion Module (AGFM)
# ===========================================================================

class AdaptiveGatedFusion1d(nn.Module):
    """
    AGFM for 1-D tensors (B, C, L).

    c_gated=1        → spatial gating: α ∈ R^{B×1×L}, broadcast over C.
    c_gated=channels → per-channel gating: α ∈ R^{B×C×L}.

    Default c_gated=1 matches the empirical finding in the paper.
    """
    def __init__(self, channels: int, c_gated: int = 1):
        super().__init__()
        GateConv = nn.Conv1d(channels * 2, c_gated, kernel_size=1)
        nn.init.constant_(GateConv.weight, 0)
        nn.init.constant_(GateConv.bias, 0)
        self.gate = nn.Sequential(GateConv, nn.Sigmoid())

    def forward(self, v_fno, v_wno):
        # v_cat: (B, 2C, L)
        cat_v = torch.cat([v_fno, v_wno], dim=1)
        # α: (B, c_gated, L) — broadcasts over C when c_gated=1
        alpha = self.gate(cat_v)
        return alpha * v_fno + (1 - alpha) * v_wno


class AdaptiveGatedFusion2d(nn.Module):
    """
    AGFM for 2-D tensors (B, C, H, W).

    c_gated=1        → spatial gating: α ∈ R^{B×1×H×W}, broadcast over C.
    c_gated=channels → per-channel gating: α ∈ R^{B×C×H×W}.

    Default c_gated=1 matches the empirical finding in the paper.
    """
    def __init__(self, channels: int, c_gated: int = 1):
        super().__init__()
        GateConv = nn.Conv2d(channels * 2, c_gated, kernel_size=1)
        nn.init.constant_(GateConv.weight, 0)
        nn.init.constant_(GateConv.bias, 0)
        self.gate = nn.Sequential(GateConv, nn.Sigmoid())

    def forward(self, v_fno, v_wno):
        # v_cat: (B, 2C, H, W)
        cat_v = torch.cat([v_fno, v_wno], dim=1)
        # α: (B, c_gated, H, W) — broadcasts over C when c_gated=1
        alpha = self.gate(cat_v)
        return alpha * v_fno + (1 - alpha) * v_wno


# ===========================================================================
# Full models
# ===========================================================================

class AWFNO1dFinalAGFM(BaseModel):
    """
    AW-FNO 1D with a single AGFM applied once after both branches finish.

    Forward pass:
        v_0   = Lifting( PosEmbed(a) )
        v_FNO = FNOBlock_T( ... FNOBlock_1(v_0) ... )
        v_WNO = WNOBlock_T( ... WNOBlock_1(v_0) ... )
        u(x)  = Projection( AGFM(v_FNO, v_WNO) )
    """
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
        non_linearity=F.gelu,
        padding: int = 0,
        dropout: float = 0.0,
        c_gated: int = 1,
    ):
        super().__init__()
        self.padding = padding

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
            dropout=dropout,
        )

        padded_size = [s + padding for s in size]

        # Independent T-layer FNO branch
        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(hidden_channels, hidden_channels, n_modes, non_linearity, dropout)
            for _ in range(n_layers)
        ])

        # Independent T-layer WNO branch
        self.wno_blocks = nn.ModuleList([
            WNOBlock1d(hidden_channels, hidden_channels, padded_size,
                       wno_level, wno_wavelet, non_linearity, dropout)
            for _ in range(n_layers)
        ])

        # Single AGFM — applied once after both branches complete
        self.agfm = AdaptiveGatedFusion1d(hidden_channels, c_gated=c_gated)

        projection_channels = int(hidden_channels * projection_channel_ratio)
        self.projection = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=1,
            non_linearity=non_linearity,
            dropout=dropout,
        )

    def forward(self, x):
        if self.pos_embed is not None:
            x = self.pos_embed(x)
        x = self.lifting(x)                             # v_0
        if self.padding > 0:
            x = F.pad(x, [0, self.padding])

        # FNO branch: v_0 → v_1^F → ... → v_FNO  (independent of WNO)
        v_fno = x
        for block in self.fno_blocks:
            v_fno = block(v_fno)

        # WNO branch: v_0 → v_1^W → ... → v_WNO  (independent of FNO)
        v_wno = x
        for block in self.wno_blocks:
            v_wno = block(v_wno)

        if self.padding > 0:
            v_fno = v_fno[..., :-self.padding]
            v_wno = v_wno[..., :-self.padding]

        # Single adaptive gated fusion at the very end
        v_fused = self.agfm(v_fno, v_wno)

        return self.projection(v_fused)


class AWFNO2dFinalAGFM(BaseModel):
    """
    AW-FNO 2D with a single AGFM applied once after both branches finish.

    Forward pass:
        v_0   = Lifting( PosEmbed(a) )
        v_FNO = FNOBlock_T( ... FNOBlock_1(v_0) ... )
        v_WNO = WNOBlock_T( ... WNOBlock_1(v_0) ... )
        u(x)  = Projection( AGFM(v_FNO, v_WNO) )

    Parameters
    ----------
    c_gated : int
        Output channels of the gating Conv. Use 1 for spatial gating
        (α broadcasts over C, empirically better) or hidden_channels for
        per-channel gating.
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
        non_linearity=F.gelu,
        padding: int = 0,
        dropout: float = 0.0,
        c_gated: int = 1,
    ):
        super().__init__()
        self.padding = padding

        if positional_embedding == "grid":
            self.pos_embed = GridEmbedding2D(in_channels)
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
            n_dim=2,
            non_linearity=non_linearity,
            dropout=dropout,
        )

        padded_size = [s + padding for s in size]

        # Independent T-layer FNO branch
        self.fno_blocks = nn.ModuleList([
            FNOBlock2d(hidden_channels, hidden_channels, n_modes, non_linearity, dropout)
            for _ in range(n_layers)
        ])

        # Independent T-layer WNO branch
        self.wno_blocks = nn.ModuleList([
            WNOBlock2d(hidden_channels, hidden_channels, padded_size,
                       wno_level, wno_wavelet, non_linearity, dropout)
            for _ in range(n_layers)
        ])

        # Single AGFM — applied once after both branches complete
        self.agfm = AdaptiveGatedFusion2d(hidden_channels, c_gated=c_gated)

        projection_channels = int(hidden_channels * projection_channel_ratio)
        self.projection = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=2,
            non_linearity=non_linearity,
            dropout=dropout,
        )

    def forward(self, x):
        if self.pos_embed is not None:
            x = self.pos_embed(x, batched=True)
        x = self.lifting(x)                             # v_0
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        # FNO branch: v_0 → v_1^F → ... → v_FNO  (independent of WNO)
        v_fno = x
        for block in self.fno_blocks:
            v_fno = block(v_fno)

        # WNO branch: v_0 → v_1^W → ... → v_WNO  (independent of FNO)
        v_wno = x
        for block in self.wno_blocks:
            v_wno = block(v_wno)

        if self.padding > 0:
            v_fno = v_fno[..., :-self.padding, :-self.padding]
            v_wno = v_wno[..., :-self.padding, :-self.padding]

        # Single adaptive gated fusion at the very end
        v_fused = self.agfm(v_fno, v_wno)

        return self.projection(v_fused)
