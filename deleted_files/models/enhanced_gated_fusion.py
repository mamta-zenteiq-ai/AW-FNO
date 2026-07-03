"""
Enhanced Gated Fusion modules for AW-FNO v2.

Three improved fusion variants over the baseline GatedFusion:

  1. DualGatedFusion  — removes the convex-combination constraint (α + (1-α) = 1).
                        Learns independent gates α_f and α_w for each branch so
                        each can be freely amplified or suppressed.

  2. SEGatedFusion    — augments the baseline local 1×1 gate with a global
                        Squeeze-and-Excitation channel-attention bias derived
                        from global average pooling of both branches. The gate
                        at every spatial location is shifted by a field-level
                        summary, giving it awareness of global structure (e.g.
                        where the shock is, what the overall flow looks like).

  3. CrossModalFusion — inserts a lightweight cross-branch modulation step
                        before SE-dual gating. WNO features are added to FNO
                        features (and vice versa) via zero-initialised 1×1
                        convolutions, then an independent SE-gated combination
                        is applied. This is the most expressive variant.

Initialization: all learnable gate parameters start at zero, so every variant
begins at α = 0.5 (equal blend) and diverges from there during training.

All classes share the same constructor signature (channels, [se_reduction=4])
and forward signature (v_f, v_w) → fused output, making them drop-in
replacements for the baseline GatedFusion1d / 2d / 3d classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1-D variants
# ─────────────────────────────────────────────────────────────────────────────

class DualGatedFusion1d(nn.Module):
    """
    Decoupled dual-gate fusion for 1-D fields.

    Replaces the convex combination α·v_f + (1-α)·v_w with independent gates:
        out = α_f·v_f + α_w·v_w
    so the network can amplify, suppress, or balance each branch freely.
    """
    def __init__(self, channels: int, **kwargs):
        super().__init__()
        self.gate_f = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.gate_w = nn.Conv1d(channels * 2, channels, kernel_size=1)
        for m in (self.gate_f, self.gate_w):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.norm = nn.InstanceNorm1d(channels)

    def forward(self, v_f: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([v_f, v_w], dim=1)              # (B, 2C, L)
        alpha_f = torch.sigmoid(self.gate_f(cat))        # (B, C, L)
        alpha_w = torch.sigmoid(self.gate_w(cat))        # (B, C, L)
        return self.norm(alpha_f * v_f + alpha_w * v_w)


class SEGatedFusion1d(nn.Module):
    """
    Squeeze-and-Excitation gated fusion for 1-D fields.

    Adds a global channel-attention bias to the local 1×1 gate logit:
        local_logit  = conv1x1([v_f, v_w])               (B, C, L)
        global_bias  = FC( ReLU( FC( GAP([v_f, v_w]) ))) (B, C)
        alpha        = sigmoid( local_logit + global_bias )
        out          = norm( alpha·v_f + (1-alpha)·v_w )

    The global branch is zero-initialised so training starts from the
    baseline local-only gate and unlocks global context gradually.
    """
    def __init__(self, channels: int, se_reduction: int = 4):
        super().__init__()
        mid = max(channels // se_reduction, 4)
        self.se_fc1 = nn.Linear(channels * 2, mid)
        self.se_fc2 = nn.Linear(mid, channels)
        nn.init.zeros_(self.se_fc2.weight)
        nn.init.zeros_(self.se_fc2.bias)

        self.local_gate = nn.Conv1d(channels * 2, channels, kernel_size=1)
        nn.init.zeros_(self.local_gate.weight)
        nn.init.zeros_(self.local_gate.bias)
        self.norm = nn.InstanceNorm1d(channels)

    def forward(self, v_f: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([v_f, v_w], dim=1)               # (B, 2C, L)
        # Global SE bias: (B, C)
        ctx = cat.mean(dim=-1)                            # (B, 2C)
        global_bias = self.se_fc2(F.relu(self.se_fc1(ctx)))  # (B, C)
        # Gate = local logit + global shift
        local_logit = self.local_gate(cat)                # (B, C, L)
        alpha = torch.sigmoid(local_logit + global_bias.unsqueeze(-1))
        return self.norm(alpha * v_f + (1 - alpha) * v_w)


class CrossModalFusion1d(nn.Module):
    """
    Cross-modal fusion for 1-D fields.

    Step 1 — cross-branch modulation (zero-init, unlocks during training):
        v_f_mod = v_f + cross_wf(v_w)   # WNO context informs FNO features
        v_w_mod = v_w + cross_fw(v_f)   # FNO context informs WNO features

    Step 2 — independent SE-gated combination on the modulated features:
        global_logit_f = FC_f( ReLU( FC1( GAP([v_f_mod, v_w_mod]) ) ))
        global_logit_w = FC_w( ReLU( FC1( GAP([v_f_mod, v_w_mod]) ) ))
        alpha_f = sigmoid( gate_f([v_f_mod, v_w_mod]) + global_logit_f )
        alpha_w = sigmoid( gate_w([v_f_mod, v_w_mod]) + global_logit_w )
        out = norm( alpha_f·v_f_mod + alpha_w·v_w_mod )

    Motivation for SOD: FNO learns the smooth global background; feeding it
    back to WNO helps place shocks accurately.  WNO knows shock locations;
    feeding that back to FNO reduces Gibbs ringing near discontinuities.
    """
    def __init__(self, channels: int, se_reduction: int = 4):
        super().__init__()
        # Cross-branch modulation (zero-init → identity at start)
        self.cross_wf = nn.Conv1d(channels, channels, kernel_size=1)
        self.cross_fw = nn.Conv1d(channels, channels, kernel_size=1)
        for m in (self.cross_wf, self.cross_fw):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        # Shared SE trunk, separate heads for F and W gates
        mid = max(channels // se_reduction, 4)
        self.se_fc1 = nn.Linear(channels * 2, mid)
        self.se_fc2_f = nn.Linear(mid, channels)
        self.se_fc2_w = nn.Linear(mid, channels)
        for fc in (self.se_fc2_f, self.se_fc2_w):
            nn.init.zeros_(fc.weight)
            nn.init.zeros_(fc.bias)

        # Local dual gates
        self.gate_f = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.gate_w = nn.Conv1d(channels * 2, channels, kernel_size=1)
        for m in (self.gate_f, self.gate_w):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        self.norm = nn.InstanceNorm1d(channels)

    def forward(self, v_f: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
        # Step 1: cross-branch modulation
        v_f_mod = v_f + self.cross_wf(v_w)
        v_w_mod = v_w + self.cross_fw(v_f)

        cat = torch.cat([v_f_mod, v_w_mod], dim=1)      # (B, 2C, L)

        # Step 2: SE global bias per branch
        ctx = cat.mean(dim=-1)                            # (B, 2C)
        h = F.relu(self.se_fc1(ctx))                      # (B, mid)
        se_f = self.se_fc2_f(h).unsqueeze(-1)             # (B, C, 1)
        se_w = self.se_fc2_w(h).unsqueeze(-1)             # (B, C, 1)

        # Step 3: independent gates (local + global bias)
        alpha_f = torch.sigmoid(self.gate_f(cat) + se_f)  # (B, C, L)
        alpha_w = torch.sigmoid(self.gate_w(cat) + se_w)  # (B, C, L)

        return self.norm(alpha_f * v_f_mod + alpha_w * v_w_mod)


# ─────────────────────────────────────────────────────────────────────────────
# 2-D variants  (H × W spatial dims)
# ─────────────────────────────────────────────────────────────────────────────

class DualGatedFusion2d(nn.Module):
    """Decoupled dual-gate fusion for 2-D fields."""
    def __init__(self, channels: int, **kwargs):
        super().__init__()
        self.gate_f = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.gate_w = nn.Conv2d(channels * 2, channels, kernel_size=1)
        for m in (self.gate_f, self.gate_w):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, v_f: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([v_f, v_w], dim=1)
        alpha_f = torch.sigmoid(self.gate_f(cat))
        alpha_w = torch.sigmoid(self.gate_w(cat))
        return self.norm(alpha_f * v_f + alpha_w * v_w)


class SEGatedFusion2d(nn.Module):
    """SE-gated fusion for 2-D fields."""
    def __init__(self, channels: int, se_reduction: int = 4):
        super().__init__()
        mid = max(channels // se_reduction, 4)
        self.se_fc1 = nn.Linear(channels * 2, mid)
        self.se_fc2 = nn.Linear(mid, channels)
        nn.init.zeros_(self.se_fc2.weight)
        nn.init.zeros_(self.se_fc2.bias)
        self.local_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        nn.init.zeros_(self.local_gate.weight)
        nn.init.zeros_(self.local_gate.bias)
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, v_f: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([v_f, v_w], dim=1)               # (B, 2C, H, W)
        ctx = cat.mean(dim=(-2, -1))                      # (B, 2C)
        global_bias = self.se_fc2(F.relu(self.se_fc1(ctx)))  # (B, C)
        local_logit = self.local_gate(cat)                # (B, C, H, W)
        alpha = torch.sigmoid(local_logit + global_bias[:, :, None, None])
        return self.norm(alpha * v_f + (1 - alpha) * v_w)


class CrossModalFusion2d(nn.Module):
    """Cross-modal fusion for 2-D fields."""
    def __init__(self, channels: int, se_reduction: int = 4):
        super().__init__()
        self.cross_wf = nn.Conv2d(channels, channels, kernel_size=1)
        self.cross_fw = nn.Conv2d(channels, channels, kernel_size=1)
        for m in (self.cross_wf, self.cross_fw):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        mid = max(channels // se_reduction, 4)
        self.se_fc1 = nn.Linear(channels * 2, mid)
        self.se_fc2_f = nn.Linear(mid, channels)
        self.se_fc2_w = nn.Linear(mid, channels)
        for fc in (self.se_fc2_f, self.se_fc2_w):
            nn.init.zeros_(fc.weight)
            nn.init.zeros_(fc.bias)
        self.gate_f = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.gate_w = nn.Conv2d(channels * 2, channels, kernel_size=1)
        for m in (self.gate_f, self.gate_w):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, v_f: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
        v_f_mod = v_f + self.cross_wf(v_w)
        v_w_mod = v_w + self.cross_fw(v_f)
        cat = torch.cat([v_f_mod, v_w_mod], dim=1)
        ctx = cat.mean(dim=(-2, -1))
        h = F.relu(self.se_fc1(ctx))
        se_f = self.se_fc2_f(h)[:, :, None, None]
        se_w = self.se_fc2_w(h)[:, :, None, None]
        alpha_f = torch.sigmoid(self.gate_f(cat) + se_f)
        alpha_w = torch.sigmoid(self.gate_w(cat) + se_w)
        return self.norm(alpha_f * v_f_mod + alpha_w * v_w_mod)


# ─────────────────────────────────────────────────────────────────────────────
# 3-D variants  (D × H × W spatial dims)
# ─────────────────────────────────────────────────────────────────────────────

class DualGatedFusion3d(nn.Module):
    """Decoupled dual-gate fusion for 3-D fields."""
    def __init__(self, channels: int, **kwargs):
        super().__init__()
        self.gate_f = nn.Conv3d(channels * 2, channels, kernel_size=1)
        self.gate_w = nn.Conv3d(channels * 2, channels, kernel_size=1)
        for m in (self.gate_f, self.gate_w):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.norm = nn.InstanceNorm3d(channels)

    def forward(self, v_f: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([v_f, v_w], dim=1)
        alpha_f = torch.sigmoid(self.gate_f(cat))
        alpha_w = torch.sigmoid(self.gate_w(cat))
        return self.norm(alpha_f * v_f + alpha_w * v_w)


class SEGatedFusion3d(nn.Module):
    """SE-gated fusion for 3-D fields."""
    def __init__(self, channels: int, se_reduction: int = 4):
        super().__init__()
        mid = max(channels // se_reduction, 4)
        self.se_fc1 = nn.Linear(channels * 2, mid)
        self.se_fc2 = nn.Linear(mid, channels)
        nn.init.zeros_(self.se_fc2.weight)
        nn.init.zeros_(self.se_fc2.bias)
        self.local_gate = nn.Conv3d(channels * 2, channels, kernel_size=1)
        nn.init.zeros_(self.local_gate.weight)
        nn.init.zeros_(self.local_gate.bias)
        self.norm = nn.InstanceNorm3d(channels)

    def forward(self, v_f: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([v_f, v_w], dim=1)               # (B, 2C, D, H, W)
        ctx = cat.mean(dim=(-3, -2, -1))                  # (B, 2C)
        global_bias = self.se_fc2(F.relu(self.se_fc1(ctx)))  # (B, C)
        local_logit = self.local_gate(cat)
        alpha = torch.sigmoid(local_logit + global_bias[:, :, None, None, None])
        return self.norm(alpha * v_f + (1 - alpha) * v_w)


class CrossModalFusion3d(nn.Module):
    """Cross-modal fusion for 3-D fields."""
    def __init__(self, channels: int, se_reduction: int = 4):
        super().__init__()
        self.cross_wf = nn.Conv3d(channels, channels, kernel_size=1)
        self.cross_fw = nn.Conv3d(channels, channels, kernel_size=1)
        for m in (self.cross_wf, self.cross_fw):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        mid = max(channels // se_reduction, 4)
        self.se_fc1 = nn.Linear(channels * 2, mid)
        self.se_fc2_f = nn.Linear(mid, channels)
        self.se_fc2_w = nn.Linear(mid, channels)
        for fc in (self.se_fc2_f, self.se_fc2_w):
            nn.init.zeros_(fc.weight)
            nn.init.zeros_(fc.bias)
        self.gate_f = nn.Conv3d(channels * 2, channels, kernel_size=1)
        self.gate_w = nn.Conv3d(channels * 2, channels, kernel_size=1)
        for m in (self.gate_f, self.gate_w):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.norm = nn.InstanceNorm3d(channels)

    def forward(self, v_f: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
        v_f_mod = v_f + self.cross_wf(v_w)
        v_w_mod = v_w + self.cross_fw(v_f)
        cat = torch.cat([v_f_mod, v_w_mod], dim=1)
        ctx = cat.mean(dim=(-3, -2, -1))
        h = F.relu(self.se_fc1(ctx))
        se_f = self.se_fc2_f(h)[:, :, None, None, None]
        se_w = self.se_fc2_w(h)[:, :, None, None, None]
        alpha_f = torch.sigmoid(self.gate_f(cat) + se_f)
        alpha_w = torch.sigmoid(self.gate_w(cat) + se_w)
        return self.norm(alpha_f * v_f_mod + alpha_w * v_w_mod)
