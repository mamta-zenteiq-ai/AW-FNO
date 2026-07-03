"""
Training script for AW-FNO v2 with Enhanced Gated Fusion on 1D Sod Shock Super-Resolution.

Task: Spatial super-resolution (×4) of compressible shock profiles.
Input:  Low-res (256×3) with Vx, density, pressure
Output: High-res (1024×3) reconstructed fields

Datasets: Combined Sod1 + Sod3 + Sod5 (65 total samples)
Evaluation: Per-field relative L2 errors + enhanced gate visualization

Fusion variants (set FUSION_TYPE below):
  'dual'  — DualGatedFusion:   independent α_f, α_w (no convex constraint)
  'se'    — SEGatedFusion:     local gate + SE channel-attention from GAP
  'cross' — CrossModalFusion:  cross-branch modulation then SE dual-gate (default)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys
import time
import h5py

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from awfno.models.awfno_v2_enhanced import AWFNOv2Enhanced_1d
from awfno.utils.losses import LpLoss


def _best_device():
    """Return the CUDA device with the most free memory, or CPU."""
    if not torch.cuda.is_available():
        return torch.device('cpu')
    best_gpu, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free, best_gpu = free, i
        except Exception:
            continue
    return torch.device(f'cuda:{best_gpu}')

# ─── Fusion type — change this to compare variants ───────────────────────────
FUSION_TYPE = 'cross'   # 'dual' | 'se' | 'cross'
SE_REDUCTION = 4        # bottleneck ratio for SE variants


def normalize_field(x):
    mu = x.mean(dim=(0, 2), keepdim=True)
    std = x.std(dim=(0, 2), keepdim=True)
    return (x - mu) / (std + 1e-8), mu, std


def denormalize_field(x_norm, mu, std):
    if isinstance(x_norm, np.ndarray):
        mu_np = mu.numpy() if isinstance(mu, torch.Tensor) else mu
        std_np = std.numpy() if isinstance(std, torch.Tensor) else std
        return x_norm * std_np + mu_np
    else:
        mu_t = mu if isinstance(mu, torch.Tensor) else torch.from_numpy(mu)
        std_t = std if isinstance(std, torch.Tensor) else torch.from_numpy(std)
        return x_norm * std_t + mu_t


def load_sod_data(data_root, downsample_factor=4):
    """Load and combine Sod1, Sod3, Sod5 datasets for super-resolution."""
    all_hr = []
    for sod_name in ['1D_CFD_Sod1.hdf5', '1D_CFD_Sod3.hdf5', '1D_CFD_Sod5.hdf5']:
        filepath = os.path.join(data_root, sod_name)
        print(f"Loading {sod_name}...")
        with h5py.File(filepath, 'r') as f:
            sod_data = np.stack([f['Vx'][:], f['density'][:], f['pressure'][:]], axis=1)
            all_hr.append(sod_data)
            print(f"  Loaded {sod_data.shape}")

    hr = torch.from_numpy(np.concatenate(all_hr, axis=0)).float()  # (N, 3, 1024)
    print(f"Combined HR shape: {hr.shape}")
    print(f"HR value ranges: Vx [{hr[:,0].min():.3f}, {hr[:,0].max():.3f}], "
          f"density [{hr[:,1].min():.3f}, {hr[:,1].max():.3f}], "
          f"pressure [{hr[:,2].min():.3f}, {hr[:,2].max():.3f}]")

    hr_norm, mu, std = normalize_field(hr)
    lr_norm = hr_norm[:, :, ::downsample_factor]
    print(f"LR shape after downsampling by ×{downsample_factor}: {lr_norm.shape}")
    return lr_norm, hr_norm, mu, std


def extract_gate_alpha(gate_module, v_f, v_w):
    """Extract gate weight tensors from any enhanced fusion module.

    Returns a dict with:
      'alpha_f': (C, L) channel-mean gate on the FNO branch (always present)
      'alpha_w': (C, L) channel-mean gate on the WNO branch (always present)
      'global_bias': (C,) SE global bias, or None if not applicable
    All tensors are on CPU.
    """
    import torch
    import torch.nn.functional as F
    from awfno.models.enhanced_gated_fusion import (
        DualGatedFusion1d, SEGatedFusion1d, CrossModalFusion1d,
    )

    gate_module.eval()
    with torch.no_grad():
        if isinstance(gate_module, CrossModalFusion1d):
            v_f_mod = v_f + gate_module.cross_wf(v_w)
            v_w_mod = v_w + gate_module.cross_fw(v_f)
            cat = torch.cat([v_f_mod, v_w_mod], dim=1)
            ctx = cat.mean(dim=-1)
            h = F.relu(gate_module.se_fc1(ctx))
            se_f = gate_module.se_fc2_f(h).unsqueeze(-1)   # (B, C, 1)
            se_w = gate_module.se_fc2_w(h).unsqueeze(-1)
            alpha_f = torch.sigmoid(gate_module.gate_f(cat) + se_f)
            alpha_w = torch.sigmoid(gate_module.gate_w(cat) + se_w)
            global_bias = {
                'FNO': h.squeeze(0).cpu().numpy(),
                'WNO': h.squeeze(0).cpu().numpy(),
            }

        elif isinstance(gate_module, SEGatedFusion1d):
            cat = torch.cat([v_f, v_w], dim=1)
            ctx = cat.mean(dim=-1)
            gb = gate_module.se_fc2(F.relu(gate_module.se_fc1(ctx)))   # (B, C)
            local_logit = gate_module.local_gate(cat)
            alpha = torch.sigmoid(local_logit + gb.unsqueeze(-1))
            alpha_f = alpha
            alpha_w = 1 - alpha
            global_bias = gb.squeeze(0).cpu().numpy()

        else:  # DualGatedFusion1d
            cat = torch.cat([v_f, v_w], dim=1)
            alpha_f = torch.sigmoid(gate_module.gate_f(cat))
            alpha_w = torch.sigmoid(gate_module.gate_w(cat))
            global_bias = None

    return {
        'alpha_f': alpha_f.squeeze(0).cpu().numpy(),   # (C, L)
        'alpha_w': alpha_w.squeeze(0).cpu().numpy(),
        'global_bias': global_bias,
    }


def plot_gate_visualization(gate_data, fusion_type, results_dir):
    """Plot enhanced gate diagnostics for the chosen fusion variant."""
    alpha_f = gate_data['alpha_f']   # (C, L)
    alpha_w = gate_data['alpha_w']   # (C, L)
    alpha_f_mean = alpha_f.mean(axis=0)   # (L,)
    alpha_w_mean = alpha_w.mean(axis=0)

    x_coords = np.arange(alpha_f_mean.shape[0])

    if fusion_type == 'dual':
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        ax = axes[0]
        ax.plot(x_coords, alpha_f_mean, color='steelblue', linewidth=2.5, label='α_f (FNO gate)')
        ax.fill_between(x_coords, 0, alpha_f_mean, alpha=0.25, color='steelblue')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='α = 0.5')
        ax.set_ylim([0, 1]); ax.set_xlabel('Spatial Position (LR)'); ax.set_ylabel('Gate value')
        ax.set_title('FNO Gate α_f(x)\n(higher → FNO features weighted more)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(x_coords, alpha_w_mean, color='darkorange', linewidth=2.5, label='α_w (WNO gate)')
        ax.fill_between(x_coords, 0, alpha_w_mean, alpha=0.25, color='darkorange')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='α = 0.5')
        ax.set_ylim([0, 1]); ax.set_xlabel('Spatial Position (LR)'); ax.set_ylabel('Gate value')
        ax.set_title('WNO Gate α_w(x)\n(higher → WNO features weighted more)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.suptitle('DualGatedFusion: Independent Branch Gates', fontsize=14, fontweight='bold')

    elif fusion_type == 'se':
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        ax = axes[0]
        ax.plot(x_coords, alpha_f_mean, color='purple', linewidth=2.5, label='α(x) — FNO weight')
        ax.fill_between(x_coords, 0, alpha_f_mean, alpha=0.25, color='purple')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='α = 0.5 (neutral)')
        ax.set_ylim([0, 1]); ax.set_xlabel('Spatial Position (LR)'); ax.set_ylabel('Gate α')
        ax.set_title('SEGatedFusion: Spatial Gate α(x)\n(local gate modulated by global SE attention)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1]
        gb = gate_data['global_bias']  # (C,) SE global channel bias
        if gb is not None:
            ax.bar(np.arange(len(gb)), gb, color='teal', alpha=0.7)
            ax.axhline(y=0, color='black', linewidth=0.8)
            ax.set_xlabel('Channel index'); ax.set_ylabel('SE global bias (pre-sigmoid logit)')
            ax.set_title('SE Global Bias per Channel\n(positive → FNO favoured; negative → WNO favoured)', fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.suptitle('SEGatedFusion: Local Gate + Global SE Attention', fontsize=14, fontweight='bold')

    else:  # cross
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        ax = axes[0]
        ax.plot(x_coords, alpha_f_mean, color='steelblue', linewidth=2.5, label='α_f (modulated FNO)')
        ax.fill_between(x_coords, 0, alpha_f_mean, alpha=0.25, color='steelblue')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim([0, 1]); ax.set_xlabel('Spatial Position (LR)'); ax.set_ylabel('Gate α_f')
        ax.set_title('FNO Gate α_f(x) after\nWNO→FNO cross-modulation', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(x_coords, alpha_w_mean, color='darkorange', linewidth=2.5, label='α_w (modulated WNO)')
        ax.fill_between(x_coords, 0, alpha_w_mean, alpha=0.25, color='darkorange')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim([0, 1]); ax.set_xlabel('Spatial Position (LR)'); ax.set_ylabel('Gate α_w')
        ax.set_title('WNO Gate α_w(x) after\nFNO→WNO cross-modulation', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[2]
        dominance = alpha_f_mean - alpha_w_mean
        colors = np.where(dominance > 0, 'steelblue', 'darkorange')
        ax.bar(x_coords, dominance, color=colors, alpha=0.7, width=1.0)
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_xlabel('Spatial Position (LR)'); ax.set_ylabel('α_f − α_w')
        ax.set_title('Branch Dominance: α_f − α_w\n(blue → FNO; orange → WNO dominant)', fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.suptitle('CrossModalFusion: Post-Modulation SE Dual Gates', fontsize=14, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(results_dir, f'awfno_v2_sod_{fusion_type}_gate_viz.png')
    plt.savefig(path, dpi=150)
    print(f"Enhanced gate visualization saved to {path}")
    plt.close()


def train_sod():
    # ─── 0. Reproducibility ──────────────────────────────────────────────────
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ─── 1. Configuration ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = _best_device()
    print(f"Using device: {device}")
    print(f"Fusion type : {FUSION_TYPE}")

    epochs = 500
    batch_size = 16
    learning_rate = 1e-3
    print_every = 50

    downsample_factor = 4
    lr_resolution = 1024 // downsample_factor   # 256
    hr_resolution = 1024

    data_root = '/media/HDD/mamta_backup/datasets/PDEBench/comp_ns/1d'
    results_dir = os.path.join(PROJECT_ROOT, 'results', f'awfno_v2_sod_enhanced_{FUSION_TYPE}')
    os.makedirs(results_dir, exist_ok=True)

    # ─── 2. Load Data ────────────────────────────────────────────────────────
    print("Loading Sod shock datasets...")
    lr, hr, mu, std = load_sod_data(data_root, downsample_factor=downsample_factor)

    n_samples = lr.shape[0]
    print(f"\nTotal samples: {n_samples}")

    # N-1 train / 1 test — one unseen sample held out for evaluation
    rng_split = np.random.RandomState(seed)
    indices = rng_split.permutation(n_samples)
    test_idx  = indices[:1]       # single held-out test sample
    train_idx = indices[1:]       # all remaining samples for training

    x_train, x_test = lr[train_idx], lr[test_idx]
    y_train, y_test = hr[train_idx], hr[test_idx]
    print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(x_test,  y_test),  batch_size=batch_size, shuffle=False)

    # ─── 3. Model with Upsampling Wrapper ────────────────────────────────────
    class SuperResolutionWrapper(nn.Module):
        def __init__(self, base_model, hr_size):
            super().__init__()
            self.base_model = base_model
            self.hr_size = hr_size

        def forward(self, x):
            out = self.base_model(x)
            return nn.functional.interpolate(out, size=self.hr_size, mode='linear', align_corners=True)

    base_model = AWFNOv2Enhanced_1d(
        in_channels=3,
        out_channels=3,
        n_modes=64,
        size=[lr_resolution],
        hidden_channels=64,
        n_fno_layers=2,
        n_wno_layers=4,
        wno_wavelet='db6',
        padding=0,
        dropout=0.0,
        fusion_type=FUSION_TYPE,
        se_reduction=SE_REDUCTION,
    )

    model = SuperResolutionWrapper(base_model, hr_resolution).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nAWFNOv2Enhanced_1d [{FUSION_TYPE}] — trainable parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion_l2 = nn.MSELoss()

    # ─── 4. Training Loop ────────────────────────────────────────────────────
    train_loss_history, test_loss_history = [], []
    train_loss_per_field = {0: [], 1: [], 2: []}
    test_loss_per_field  = {0: [], 1: [], 2: []}
    field_names = ['Vx', 'density', 'pressure']

    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_per_field = {0: 0.0, 1: 0.0, 2: 0.0}

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion_l2(out, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            for fi in range(3):
                train_per_field[fi] += criterion_l2(out[:, fi, :], batch_y[:, fi, :]).item()

        train_loss /= len(train_loader)
        for fi in range(3):
            train_per_field[fi] /= len(train_loader)
            train_loss_per_field[fi].append(train_per_field[fi])
        train_loss_history.append(train_loss)

        model.eval()
        test_loss = 0.0
        test_per_field = {0: 0.0, 1: 0.0, 2: 0.0}
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = model(batch_x)
                test_loss += criterion_l2(out, batch_y).item()
                for fi in range(3):
                    test_per_field[fi] += criterion_l2(out[:, fi, :], batch_y[:, fi, :]).item()

        test_loss /= len(test_loader)
        for fi in range(3):
            test_per_field[fi] /= len(test_loader)
            test_loss_per_field[fi].append(test_per_field[fi])
        test_loss_history.append(test_loss)
        scheduler.step()

        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Train L2: {train_loss:.6f} | Test L2: {test_loss:.6f}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")

    # ─── 5. Evaluation: Per-field Relative L2 ────────────────────────────────
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            all_pred.append(model(batch_x).cpu())
            all_true.append(batch_y.cpu())

    pred_total = torch.cat(all_pred, dim=0)
    true_total = torch.cat(all_true, dim=0)

    print("\n" + "=" * 60)
    print("PER-FIELD RELATIVE L2 ERRORS (NORMALIZED)")
    print("=" * 60)
    rel_l2_per_field = {}
    for fi, fname in enumerate(field_names):
        rel_l2 = torch.norm(pred_total[:, fi, :] - true_total[:, fi, :]) / torch.norm(true_total[:, fi, :])
        rel_l2_per_field[fi] = rel_l2.item()
        print(f"{fname:12s}: {rel_l2:.6f}")
    print("=" * 60)

    # ─── 6. Plot Loss ─────────────────────────────────────────────────────────
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(train_loss_history, label='Train', linewidth=2)
    plt.plot(test_loss_history,  label='Test',  linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('L2 Loss')
    plt.title(f'Overall L2 Loss [{FUSION_TYPE}]', fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    for fi, fname in enumerate(field_names):
        plt.plot(train_loss_per_field[fi], label=fname, linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('L2 Loss')
    plt.title('Per-Field Training Loss', fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    for fi, fname in enumerate(field_names):
        plt.plot(test_loss_per_field[fi], label=fname, linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('L2 Loss')
    plt.title('Per-Field Test Loss', fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    loss_ratio = np.array(test_loss_history) / (np.array(train_loss_history) + 1e-8)
    plt.plot(loss_ratio, color='red', linewidth=2, label='Test/Train Ratio')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No overfitting')
    plt.xlabel('Epoch'); plt.ylabel('Ratio')
    plt.title('Overfitting Indicator (Test/Train)', fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'awfno_v2_sod_{FUSION_TYPE}_training_loss.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nTraining loss plot saved to {plot_path}")
    plt.close()

    # ─── 7. Shock Profile Visualisation ──────────────────────────────────────
    n_viz = min(4, len(test_idx))
    sample_indices = np.random.choice(len(test_idx), n_viz, replace=False)

    fig, axes = plt.subplots(n_viz, 3, figsize=(15, 4 * n_viz))
    if n_viz == 1:
        axes = axes.reshape(1, -1)

    for sn, si in enumerate(sample_indices):
        pred_denorm = denormalize_field(
            pred_total[si].numpy().T.reshape(1, 3, -1),
            mu.reshape(1, 3, 1), std.reshape(1, 3, 1),
        ).squeeze()
        true_denorm = denormalize_field(
            true_total[si].numpy().T.reshape(1, 3, -1),
            mu.reshape(1, 3, 1), std.reshape(1, 3, 1),
        ).squeeze()

        x_hr = np.arange(hr_resolution)
        for fi, fname in enumerate(field_names):
            ax = axes[sn, fi]
            ax.plot(x_hr, true_denorm[fi], 'b-',  linewidth=2.5, label='Ground Truth', alpha=0.8)
            ax.plot(x_hr, pred_denorm[fi], 'r--', linewidth=2.0,
                    label=f'AW-FNO v2 [{FUSION_TYPE}]', alpha=0.8)
            ax.set_xlabel('Spatial Position'); ax.set_ylabel(fname)
            ax.set_title(f'Sample {si+1} - {fname}', fontweight='bold')
            ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    shock_path = os.path.join(results_dir, f'awfno_v2_sod_{FUSION_TYPE}_shock_profiles.png')
    plt.savefig(shock_path, dpi=150)
    print(f"Shock profile visualization saved to {shock_path}")
    plt.close()

    # ─── 8. Enhanced Gate Visualisation ──────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"GATE VISUALIZATION [{FUSION_TYPE.upper()}]")
    print("=" * 60)

    sample_idx = sample_indices[0]
    sample_x = x_test[sample_idx:sample_idx + 1].to(device)
    m = model.base_model

    with torch.no_grad():
        x = sample_x
        if m.pos_embed:
            x = m.pos_embed(x)
        x = m.lifting(x)
        res = x
        if m.padding > 0:
            x = torch.nn.functional.pad(x, [0, m.padding])
        v_f = m.fourier_branch(x)
        v_w = m.wavelet_branch(x)

    gate_data = extract_gate_alpha(m.gate, v_f, v_w)
    plot_gate_visualization(gate_data, FUSION_TYPE, results_dir)

    alpha_f_mean = gate_data['alpha_f'].mean(axis=0)
    alpha_w_mean = gate_data['alpha_w'].mean(axis=0)
    print(f"α_f: mean={alpha_f_mean.mean():.4f}  range=[{alpha_f_mean.min():.4f}, {alpha_f_mean.max():.4f}]")
    print(f"α_w: mean={alpha_w_mean.mean():.4f}  range=[{alpha_w_mean.min():.4f}, {alpha_w_mean.max():.4f}]")

    # ─── 9. Save Model ────────────────────────────────────────────────────────
    model_path = os.path.join(results_dir, f'awfno_v2_sod_{FUSION_TYPE}_best.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    import json
    # Store per-field errors with field-name keys so compare_results.py can read them
    rel_l2_named = {field_names[fi]: v for fi, v in rel_l2_per_field.items()}
    metadata = {
        'model_name': f'awfno_v2_{FUSION_TYPE}',
        'fusion_type': FUSION_TYPE,
        'se_reduction': SE_REDUCTION,
        'downsample_factor': downsample_factor,
        'lr_resolution': lr_resolution,
        'hr_resolution': hr_resolution,
        'n_params': n_params,
        'rel_l2_per_field': rel_l2_named,
        'rel_l2_mean': float(sum(rel_l2_named.values()) / len(rel_l2_named)),
        'epochs_trained': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'architecture': 'AWFNOv2Enhanced',
    }
    with open(os.path.join(results_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {os.path.join(results_dir, 'metadata.json')}")


if __name__ == "__main__":
    train_sod()
