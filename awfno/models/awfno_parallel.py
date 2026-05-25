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
# PARALLELISM STRATEGY
# ---------------------------------------------------------------------------
# torch.jit.fork(fn, *args) submits fn(*args) to a background thread and
# returns a Future immediately — the caller does NOT block.
# torch.jit.wait(future) blocks until that Future's result is ready.
#
# Inside one AWFNO block, fno_conv, wno_conv and skip all read the same
# input x and do not depend on each other, so they can be forked together.
# PyTorch's CUDA runtime can then overlap their kernel execution across
# multiple CUDA streams on the same device.
#
# IMPORTANT — single-device vs dual-GPU:
#   torch.jit.fork runs on the SAME device the tensors live on (e.g. cuda:0).
#   It does NOT spread work across multiple GPUs automatically.
#   To use BOTH GPUs on this pascal server you need explicit model parallelism:
#     - place self.fno_conv parameters on cuda:0
#     - place self.wno_conv parameters on cuda:1
#     - inside forward, move x to cuda:1 for wno_conv, then move the result
#       back to cuda:0 before fusing.
#   See AWFNOBlock2dDualGPU at the bottom of this file for that pattern.
#   Note: the dual-GPU version incurs a PCIe transfer each forward pass, so
#   it only pays off if wno_conv is the dominant compute bottleneck.
# ---------------------------------------------------------------------------


class AdaptiveGatedFusion1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        GateConv = nn.Conv1d(channels * 2, channels, kernel_size=1)
        nn.init.constant_(GateConv.weight, 0)
        nn.init.constant_(GateConv.bias, 0)
        self.gate = nn.Sequential(GateConv, nn.Sigmoid())

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
    ):
        super().__init__()
        self.fno_conv = SpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            bias=False
        )
        self.wno_conv = WaveConv1d(
            in_channels, out_channels, wno_level, wno_size, wavelet=wno_wavelet
        )
        self.skip = skip_connection(
            in_features=in_channels,
            out_features=out_channels,
            skip_type="linear",
            n_dim=1,
            bias=True
        )
        self.gfm = AdaptiveGatedFusion1d(out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.non_linearity = non_linearity
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        # --- PARALLEL LAUNCH ---
        # torch.jit.fork dispatches each call to its own thread immediately
        # and returns a lightweight Future object — no waiting yet.
        # All three branches read x and are completely independent of each
        # other, so they are safe to run concurrently.
        future_fno  = torch.jit.fork(self.fno_conv, x)  # FFT → complex multiply → iFFT
        future_wno  = torch.jit.fork(self.wno_conv, x)  # DWT → multiply → iDWT
        future_skip = torch.jit.fork(self.skip,     x)  # pointwise linear projection

        # --- SYNCHRONISE ---
        # torch.jit.wait blocks until the forked call has finished and
        # returns its result.  Waiting on the three futures in order lets
        # the CUDA scheduler overlap their kernel streams.
        v_fno  = torch.jit.wait(future_fno)
        v_wno  = torch.jit.wait(future_wno)
        x_skip = torch.jit.wait(future_skip)

        # Adaptive gated fusion: α * v_fno + (1-α) * v_wno
        v_gated = self.gfm(v_fno, v_wno)

        out = v_gated + x_skip
        out = out.permute(0, 2, 1)   # (B, C, L) -> (B, L, C) for LayerNorm
        out = self.norm(out)
        out = out.permute(0, 2, 1)   # (B, L, C) -> (B, C, L)

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
        norm: Optional[str] = "layer_norm"
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
        for _ in range(n_layers):
            blocks.append(AWFNOBlock1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=n_modes,
                wno_size=padded_size,
                wno_level=wno_level,
                wno_wavelet=wno_wavelet,
                non_linearity=non_linearity,
                dropout=dropout
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


# ===========================================================================
# 2-D variants
# ===========================================================================

class AdaptiveGatedFusion2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        GateConv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        nn.init.constant_(GateConv.weight, 0)
        nn.init.constant_(GateConv.bias, 0)
        self.gate = nn.Sequential(GateConv, nn.Sigmoid())

    def forward(self, v_fno, v_wno):
        cat_v = torch.cat([v_fno, v_wno], dim=1)
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
    ):
        super().__init__()
        self.fno_conv = SpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            bias=False
        )
        self.wno_conv = WaveConv2d(
            in_channels, out_channels, wno_level, wno_size, wavelet=wno_wavelet
        )
        self.skip = skip_connection(
            in_features=in_channels,
            out_features=out_channels,
            skip_type="linear",
            n_dim=2,
            bias=True
        )
        self.gfm = AdaptiveGatedFusion2d(out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.non_linearity = non_linearity
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        # --- PARALLEL LAUNCH ---
        # Each fork returns immediately; the three CUDA kernel chains
        # (FFT path, DWT path, linear skip) are submitted concurrently
        # and the GPU scheduler interleaves them as SM capacity allows.
        future_fno  = torch.jit.fork(self.fno_conv, x)
        future_wno  = torch.jit.fork(self.wno_conv, x)
        future_skip = torch.jit.fork(self.skip,     x)

        # --- SYNCHRONISE ---
        v_fno  = torch.jit.wait(future_fno)
        v_wno  = torch.jit.wait(future_wno)
        x_skip = torch.jit.wait(future_skip)

        v_gated = self.gfm(v_fno, v_wno)

        out = v_gated + x_skip
        out = out.permute(0, 2, 3, 1)   # (B,C,H,W) -> (B,H,W,C) for LayerNorm
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)   # (B,H,W,C) -> (B,C,H,W)

        out = self.non_linearity(out)
        return self.dropout(out)


class AWFNO2d(BaseModel):
    """
    Parallel AWFNO 2D: fno_conv, wno_conv and skip run concurrently inside
    each block via torch.jit.fork, overlapping CUDA kernels on a single GPU.
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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.n_modes = n_modes
        self.size = size

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
            dropout=dropout
        )

        padded_size = [s + padding for s in size]

        blocks = []
        for _ in range(n_layers):
            blocks.append(AWFNOBlock2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=n_modes,
                wno_size=padded_size,
                wno_level=wno_level,
                wno_wavelet=wno_wavelet,
                non_linearity=non_linearity,
                dropout=dropout
            ))
        self.blocks = nn.ModuleList(blocks)

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


# ===========================================================================
# DUAL-GPU variant — uses BOTH GPUs on this pascal server
# ===========================================================================
# fno_conv lives on fno_device (default cuda:0)
# wno_conv lives on wno_device (default cuda:1)
# skip     lives on fno_device
#
# Inside forward:
#   1. x stays on fno_device for fno_conv and skip (both forked).
#   2. x is moved to wno_device for wno_conv (also forked).
#   3. After synchronising, v_wno is moved back to fno_device.
#   4. Fusion happens entirely on fno_device.
#
# TRADE-OFF: every forward pass pays a PCIe transfer for the wno input and
# wno output.  This only wins over single-GPU parallel-streams if wno_conv
# is the throughput bottleneck AND transfer latency is smaller than the
# saved compute time.  Profile with torch.profiler before switching.
# ===========================================================================

class AWFNOBlock2dDualGPU(nn.Module):
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
        fno_device: str = 'cuda:0',
        wno_device: str = 'cuda:1',
    ):
        super().__init__()
        self.fno_device = torch.device(fno_device)
        self.wno_device = torch.device(wno_device)

        # fno_conv and skip are pinned to fno_device
        self.fno_conv = SpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            bias=False
        ).to(self.fno_device)
        self.skip = skip_connection(
            in_features=in_channels,
            out_features=out_channels,
            skip_type="linear",
            n_dim=2,
            bias=True
        ).to(self.fno_device)

        # wno_conv is pinned to wno_device
        self.wno_conv = WaveConv2d(
            in_channels, out_channels, wno_level, wno_size, wavelet=wno_wavelet
        ).to(self.wno_device)

        # Everything below runs on fno_device
        self.gfm  = AdaptiveGatedFusion2d(out_channels).to(self.fno_device)
        self.norm = nn.LayerNorm(out_channels).to(self.fno_device)
        self.non_linearity = non_linearity
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        # x is already on fno_device (cuda:0)

        # Move x to wno_device for the wavelet branch — non-blocking transfer
        # so the CPU thread can immediately proceed to launch fno_conv/skip
        x_wno = x.to(self.wno_device, non_blocking=True)

        # Launch fno_conv and skip on fno_device in parallel
        future_fno  = torch.jit.fork(self.fno_conv, x)
        future_skip = torch.jit.fork(self.skip,     x)

        # Launch wno_conv on wno_device (runs on cuda:1 independently)
        future_wno  = torch.jit.fork(self.wno_conv, x_wno)

        # Collect results
        v_fno  = torch.jit.wait(future_fno)
        x_skip = torch.jit.wait(future_skip)
        v_wno  = torch.jit.wait(future_wno)

        # Move wno result back to fno_device for fusion
        v_wno = v_wno.to(self.fno_device, non_blocking=True)

        v_gated = self.gfm(v_fno, v_wno)
        out = v_gated + x_skip
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)
        out = self.non_linearity(out)
        return self.dropout(out)


class AWFNO2dDualGPU(BaseModel):
    """
    Dual-GPU AWFNO: fno_conv on cuda:0, wno_conv on cuda:1.
    All other sub-modules (lifting, skip, gfm, projection) stay on cuda:0.
    The input tensor must be on cuda:0; output is also on cuda:0.
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
        fno_device: str = 'cuda:0',
        wno_device: str = 'cuda:1',
    ):
        super().__init__()
        self.fno_device = torch.device(fno_device)
        self.padding = padding
        self.n_modes = n_modes
        self.size = size

        if positional_embedding == "grid":
            self.pos_embed = GridEmbedding2D(in_channels).to(self.fno_device)
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
            dropout=dropout
        ).to(self.fno_device)

        padded_size = [s + padding for s in size]

        blocks = []
        for _ in range(n_layers):
            blocks.append(AWFNOBlock2dDualGPU(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=n_modes,
                wno_size=padded_size,
                wno_level=wno_level,
                wno_wavelet=wno_wavelet,
                non_linearity=non_linearity,
                dropout=dropout,
                fno_device=fno_device,
                wno_device=wno_device,
            ))
        self.blocks = nn.ModuleList(blocks)

        projection_channels = int(hidden_channels * projection_channel_ratio)
        self.projection = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=2,
            non_linearity=non_linearity,
            dropout=dropout
        ).to(self.fno_device)

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
