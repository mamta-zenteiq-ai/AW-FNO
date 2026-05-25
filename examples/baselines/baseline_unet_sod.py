"""
U-Net baseline for SOD shock super-resolution.

Implements a standard 1-D encoder-decoder U-Net with skip connections.
The model operates at LR resolution (256) and is wrapped with the same
SuperResolutionWrapper (×4 linear interpolation) used by all baselines.

Architecture (base_channels=32, bc=32):
  Encoder:
    stem:        (B,  3, 256) → (B, bc,   256)   s0 — skip to dec1
    pool → enc1: (B, bc,  128) → (B, 2bc, 128)   s1 — skip to dec2
    pool → enc2: (B, 2bc,  64) → (B, 4bc,  64)   s2 — skip to dec3
  Bottleneck:
    pool → btn:  (B, 4bc,  32) → (B, 8bc,  32)   deepest features
  Decoder:
    up3 + cat(s2) → dec3: (B, 4bc,  64)
    up2 + cat(s1) → dec2: (B, 2bc, 128)
    up1 + cat(s0) → dec1: (B,  bc, 256)
  Head:  (B, bc, 256) → (B, 3, 256)
  → ×4 interpolation → (B, 3, 1024)
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from sod_common import (
    DATA_ROOT, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    run_experiment,
)

MODEL_NAME  = 'unet_baseline'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'baselines', MODEL_NAME)

BASE_CHANNELS = 32    # progressive doubling: 32→64→128→256


# ─── Building blocks ──────────────────────────────────────────────────────────

class ConvBlock1d(nn.Module):
    """Two Conv1d(k=3, pad=1) layers with GELU activation."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class Up1d(nn.Module):
    """Bilinear ×2 upsampling followed by channel reduction Conv1d."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(self.up(x))


# ─── U-Net ────────────────────────────────────────────────────────────────────

class UNet1d(nn.Module):
    """1-D U-Net for signal super-resolution / regression."""
    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base_channels: int = BASE_CHANNELS):
        super().__init__()
        bc = base_channels

        # Encoder
        self.stem = ConvBlock1d(in_channels, bc)       # (B, bc,   256) → s0
        self.enc1 = ConvBlock1d(bc,   bc * 2)          # (B, 2bc,  128) → s1
        self.enc2 = ConvBlock1d(bc*2, bc * 4)          # (B, 4bc,   64) → s2
        self.pool = nn.MaxPool1d(2)

        # Bottleneck — pool one more time then process
        self.bottleneck = ConvBlock1d(bc * 4, bc * 8)  # (B, 8bc,   32)

        # Decoder — Up halves channels; skip cat doubles them back
        self.up3  = Up1d(bc * 8, bc * 4)               # (B, 4bc,   64)
        self.dec3 = ConvBlock1d(bc * 8, bc * 4)        # cat(4bc+4bc) → 4bc
        self.up2  = Up1d(bc * 4, bc * 2)               # (B, 2bc,  128)
        self.dec2 = ConvBlock1d(bc * 4, bc * 2)        # cat(2bc+2bc) → 2bc
        self.up1  = Up1d(bc * 2, bc)                   # (B,  bc,  256)
        self.dec1 = ConvBlock1d(bc * 2, bc)            # cat(bc+bc) → bc

        # Output
        self.head = nn.Conv1d(bc, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        s0 = self.stem(x)                    # (B, bc,   256)
        s1 = self.enc1(self.pool(s0))        # (B, 2bc,  128)
        s2 = self.enc2(self.pool(s1))        # (B, 4bc,   64)

        # Bottleneck
        b  = self.bottleneck(self.pool(s2))  # (B, 8bc,   32)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b),  s2], dim=1))  # (B, 4bc,  64)
        d2 = self.dec2(torch.cat([self.up2(d3), s1], dim=1))  # (B, 2bc, 128)
        d1 = self.dec1(torch.cat([self.up1(d2), s0], dim=1))  # (B,  bc, 256)

        return self.head(d1)                 # (B, out_channels, 256)


if __name__ == '__main__':
    model = UNet1d(in_channels=3, out_channels=3, base_channels=BASE_CHANNELS)
    run_experiment(
        model_name=MODEL_NAME,
        base_model=model,
        results_dir=RESULTS_DIR,
        data_root=DATA_ROOT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        extra_meta={
            'architecture': 'UNet1d',
            'base_channels': BASE_CHANNELS,
        },
    )
