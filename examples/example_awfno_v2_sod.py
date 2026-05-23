"""
Training script for AW-FNO v2 on 1D Sod Shock Super-Resolution Task.

Task: Spatial super-resolution (×4) of compressible shock profiles.
Input:  Low-res (256×3) with Vx, density, pressure
Output: High-res (1024×3) reconstructed fields

Datasets: Combined Sod1 + Sod3 + Sod5 (65 total samples)
Evaluation: Per-field relative L2 errors + gate visualization at shock location
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

from awfno.models.awfno_v2 import AWFNOv2_1d
from awfno.utils.losses import LpLoss


def normalize_field(x):
    """Per-field normalization using z-score normalization.
    
    Args:
        x: tensor of shape (N_samples, 3, spatial_dim)
    
    Returns:
        x_norm: normalized tensor
        mu: mean per field (1, 3, 1)
        std: standard deviation per field (1, 3, 1)
    """
    mu = x.mean(dim=(0, 2), keepdim=True)
    std = x.std(dim=(0, 2), keepdim=True)
    x_norm = (x - mu) / (std + 1e-8)
    return x_norm, mu, std


def denormalize_field(x_norm, mu, std):
    """Denormalize field using stored statistics."""
    # Handle both numpy and torch inputs
    if isinstance(x_norm, np.ndarray):
        mu_np = mu.numpy() if isinstance(mu, torch.Tensor) else mu
        std_np = std.numpy() if isinstance(std, torch.Tensor) else std
        return x_norm * std_np + mu_np
    else:
        mu_t = mu if isinstance(mu, torch.Tensor) else torch.from_numpy(mu)
        std_t = std if isinstance(std, torch.Tensor) else torch.from_numpy(std)
        return x_norm * std_t + mu_t


def load_sod_data(data_root, downsample_factor=4):
    """Load and combine Sod1, Sod3, Sod5 datasets for super-resolution.
    
    Args:
        data_root: path to PDEBench comp_ns/1d directory
        downsample_factor: downsampling factor (4 for 1024→256)
    
    Returns:
        lr: low-resolution input (N, 3, 256)
        hr: high-resolution ground truth (N, 3, 1024)
        mu: per-field normalization mean
        std: per-field normalization std
    """
    all_hr = []
    
    # Load Sod1, Sod3, Sod5
    for sod_name in ['1D_CFD_Sod1.hdf5', '1D_CFD_Sod3.hdf5', '1D_CFD_Sod5.hdf5']:
        filepath = os.path.join(data_root, sod_name)
        print(f"Loading {sod_name}...")
        
        with h5py.File(filepath, 'r') as f:
            # Extract fields: (N_samples, 1024) for each
            vx = f['Vx'][:]
            density = f['density'][:]
            pressure = f['pressure'][:]
            
            # Stack fields → (N_samples, 3, 1024)
            # Order: [Vx, density, pressure]
            sod_data = np.stack([vx, density, pressure], axis=1)
            all_hr.append(sod_data)
            print(f"  Loaded {sod_data.shape}")
    
    # Concatenate all datasets
    hr = np.concatenate(all_hr, axis=0)  # (N_total, 3, 1024)
    hr = torch.from_numpy(hr).float()
    
    print(f"Combined HR shape: {hr.shape}")
    print(f"HR value ranges: Vx [{hr[:,0].min():.3f}, {hr[:,0].max():.3f}], "
          f"density [{hr[:,1].min():.3f}, {hr[:,1].max():.3f}], "
          f"pressure [{hr[:,2].min():.3f}, {hr[:,2].max():.3f}]")
    
    # Per-field normalization BEFORE downsampling
    hr_norm, mu, std = normalize_field(hr)
    
    # Downsample using strided subsampling (preserves sharp shocks)
    lr_norm = hr_norm[:, :, ::downsample_factor]
    
    print(f"LR shape after downsampling by ×{downsample_factor}: {lr_norm.shape}")
    
    return lr_norm, hr_norm, mu, std


def train_sod():
    # ─── 0. Reproducibility ──────────────────────────────────────────
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ─── 1. Configuration ────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    epochs = 500
    batch_size = 16
    learning_rate = 1e-3
    print_every = 50
    
    downsample_factor = 4  # ×4 super-resolution
    lr_resolution = 1024 // downsample_factor  # 256
    hr_resolution = 1024

    data_root = '/media/HDD/mamta_backup/datasets/PDEBench/comp_ns/1d'
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'awfno_v2_sod')
    os.makedirs(results_dir, exist_ok=True)

    # ─── 2. Load Data ────────────────────────────────────────────────
    print("Loading Sod shock datasets...")
    lr, hr, mu, std = load_sod_data(data_root, downsample_factor=downsample_factor)
    
    n_samples = lr.shape[0]
    print(f"\nTotal samples: {n_samples}")
    
    # 80/20 train/test split
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    x_train, x_test = lr[train_idx], lr[test_idx]
    y_train, y_test = hr[train_idx], hr[test_idx]
    
    print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size, shuffle=False,
    )

    # ─── 3. Model with Upsampling Wrapper ───────────────────────────
    class SuperResolutionWrapper(nn.Module):
        """Wraps AWFNOv2_1d to upsample output from LR to HR resolution."""
        def __init__(self, base_model, hr_size):
            super().__init__()
            self.base_model = base_model
            self.hr_size = hr_size
        
        def forward(self, x):
            out = self.base_model(x)
            # Upsample from LR (256) to HR (1024)
            out_upsampled = torch.nn.functional.interpolate(
                out, size=self.hr_size, mode='linear', align_corners=True
            )
            return out_upsampled
    
    base_model = AWFNOv2_1d(
        in_channels=3,          # [Vx, density, pressure]
        out_channels=3,
        n_modes=64,             # Fourier modes for 1D
        size=[lr_resolution],   # (256,) for input
        hidden_channels=64,
        n_fno_layers=0,
        n_wno_layers=8,
        wno_wavelet='db6',
        padding=0,
        dropout=0.0,
    )
    
    model = SuperResolutionWrapper(base_model, hr_resolution).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nAWFNOv2_1d — trainable parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    criterion_l2 = nn.MSELoss()

    # ─── 4. Training Loop ────────────────────────────────────────────
    train_loss_history = []
    test_loss_history = []
    
    # Per-field loss history
    train_loss_per_field = {0: [], 1: [], 2: []}
    test_loss_per_field = {0: [], 1: [], 2: []}
    field_names = ['Vx', 'density', 'pressure']

    print(f"\nStarting AW-FNO v2 training on Sod shocks for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_per_field = {0: 0.0, 1: 0.0, 2: 0.0}

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            out = model(batch_x)

            # Overall L2 loss
            loss = criterion_l2(out, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # Per-field loss
            for field_idx in range(3):
                field_loss = criterion_l2(out[:, field_idx, :], batch_y[:, field_idx, :]).item()
                train_per_field[field_idx] += field_loss

        train_loss /= len(train_loader)
        for field_idx in range(3):
            train_per_field[field_idx] /= len(train_loader)
            train_loss_per_field[field_idx].append(train_per_field[field_idx])
        train_loss_history.append(train_loss)

        # Validation
        model.eval()
        test_loss = 0.0
        test_per_field = {0: 0.0, 1: 0.0, 2: 0.0}
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = model(batch_x)
                
                test_loss += criterion_l2(out, batch_y).item()
                for field_idx in range(3):
                    field_loss = criterion_l2(out[:, field_idx, :], batch_y[:, field_idx, :]).item()
                    test_per_field[field_idx] += field_loss

        test_loss /= len(test_loader)
        for field_idx in range(3):
            test_per_field[field_idx] /= len(test_loader)
            test_loss_per_field[field_idx].append(test_per_field[field_idx])
        test_loss_history.append(test_loss)

        scheduler.step()

        if epoch % print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train L2: {train_loss:.6f} | "
                f"Test L2: {test_loss:.6f}"
            )

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")

    # ─── 5. Evaluation: Per-field Relative L2 Errors ────────────────────
    model.eval()
    all_pred = []
    all_true = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            all_pred.append(pred.cpu())
            all_true.append(batch_y.cpu())
    
    pred_total = torch.cat(all_pred, dim=0)
    true_total = torch.cat(all_true, dim=0)
    
    # Compute per-field relative L2
    print("\n" + "="*60)
    print("PER-FIELD RELATIVE L2 ERRORS (NORMALIZED)")
    print("="*60)
    
    rel_l2_per_field = {}
    for field_idx, field_name in enumerate(field_names):
        pred_field = pred_total[:, field_idx, :]
        true_field = true_total[:, field_idx, :]
        
        rel_l2 = torch.norm(pred_field - true_field) / torch.norm(true_field)
        rel_l2_per_field[field_idx] = rel_l2.item()
        
        print(f"{field_name:12s}: {rel_l2:.6f}")
    
    print("="*60)

    # ─── 6. Plot Loss ────────────────────────────────────────────────
    plt.figure(figsize=(14, 10))
    
    # Overall loss
    plt.subplot(2, 2, 1)
    plt.plot(train_loss_history, label='Train', linewidth=2)
    plt.plot(test_loss_history, label='Test', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('L2 Loss', fontsize=12)
    plt.title('Overall L2 Loss', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Per-field training loss
    plt.subplot(2, 2, 2)
    for field_idx, field_name in enumerate(field_names):
        plt.plot(train_loss_per_field[field_idx], label=field_name, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('L2 Loss', fontsize=12)
    plt.title('Per-Field Training Loss', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Per-field test loss
    plt.subplot(2, 2, 3)
    for field_idx, field_name in enumerate(field_names):
        plt.plot(test_loss_per_field[field_idx], label=field_name, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('L2 Loss', fontsize=12)
    plt.title('Per-Field Test Loss', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Loss ratio (Test / Train) to detect overfitting
    plt.subplot(2, 2, 4)
    loss_ratio = np.array(test_loss_history) / (np.array(train_loss_history) + 1e-8)
    plt.plot(loss_ratio, color='red', linewidth=2, label='Test/Train Ratio')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No overfitting')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Ratio', fontsize=12)
    plt.title('Overfitting Indicator (Test/Train)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'awfno_v2_sod_training_loss.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nTraining loss plot saved to {plot_path}")
    plt.close()

    # ─── 7. Visualization: Shock Profiles ────────────────────────────
    n_samples_viz = min(4, len(test_idx))
    sample_indices = np.random.choice(len(test_idx), n_samples_viz, replace=False)
    
    fig, axes = plt.subplots(n_samples_viz, 3, figsize=(15, 4*n_samples_viz))
    if n_samples_viz == 1:
        axes = axes.reshape(1, -1)
    
    for sample_num, sample_idx in enumerate(sample_indices):
        pred_norm = pred_total[sample_idx].numpy()
        true_norm = true_total[sample_idx].numpy()
        
        # Denormalize for visualization
        pred_denorm = denormalize_field(
            pred_norm.T.reshape(1, 3, -1),
            mu.reshape(1, 3, 1),
            std.reshape(1, 3, 1)
        ).squeeze()
        
        true_denorm = denormalize_field(
            true_norm.T.reshape(1, 3, -1),
            mu.reshape(1, 3, 1),
            std.reshape(1, 3, 1)
        ).squeeze()
        
        for field_idx, field_name in enumerate(field_names):
            ax = axes[sample_num, field_idx]
            
            x_hr = np.arange(hr_resolution)
            
            ax.plot(x_hr, true_denorm[field_idx], 'b-', linewidth=2.5, label='Ground Truth', alpha=0.8)
            ax.plot(x_hr, pred_denorm[field_idx], 'r--', linewidth=2, label='AW-FNO v2', alpha=0.8)
            
            ax.set_xlabel('Spatial Position', fontsize=11)
            ax.set_ylabel(field_name, fontsize=11)
            ax.set_title(f'Sample {sample_idx+1} - {field_name}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    shock_plot_path = os.path.join(results_dir, 'awfno_v2_sod_shock_profiles.png')
    plt.savefig(shock_plot_path, dpi=150)
    print(f"Shock profile visualization saved to {shock_plot_path}")
    plt.close()

    # ─── 8. Gate Visualization: Alpha map at shock location ────────────
    print("\n" + "="*60)
    print("GATE VISUALIZATION: Recording α(x) at shock location")
    print("="*60)
    
    # Extract gate values from base model
    sample_idx = sample_indices[0]
    sample_x = x_test[sample_idx:sample_idx+1].to(device)
    sample_y = y_test[sample_idx:sample_idx+1].to(device)
    
    # Access base model from wrapper
    base_model_ref = model.base_model
    
    # Forward pass with intermediate activation capture
    with torch.no_grad():
        # We need to modify forward to capture gate outputs
        # For now, we'll do a custom forward pass
        x = sample_x
        
        # Apply lifting
        if base_model_ref.pos_embed:
            x = base_model_ref.pos_embed(x)
        x = base_model_ref.lifting(x)
        res = x
        
        if base_model_ref.padding > 0:
            x = torch.nn.functional.pad(x, [0, base_model_ref.padding])
        
        v_f = base_model_ref.fourier_branch(x)
        v_w = base_model_ref.wavelet_branch(x)
        
        # Capture gate output (alpha)
        gate_input = torch.cat([v_f, v_w], dim=1)
        alpha = base_model_ref.gate.sigmoid(base_model_ref.gate.gate_conv(gate_input))
        
        # alpha shape: (1, hidden_channels, 256)
        alpha_mean = alpha.mean(dim=1).squeeze().cpu().numpy()  # Average over channels
    
    # Plot gate visualization
    fig, ax = plt.subplots(figsize=(14, 5))
    
    x_coords = np.arange(len(alpha_mean))
    ax.plot(x_coords, alpha_mean, linewidth=2.5, color='purple', label='Gate α(x)')
    ax.fill_between(x_coords, 0, alpha_mean, alpha=0.3, color='purple')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='α = 0.5 (neutral)')
    ax.set_xlabel('Spatial Position (LR)', fontsize=12)
    ax.set_ylabel('Gate α (FNO weight)', fontsize=12)
    ax.set_title('Gate Visualization: α(x) at Shock Location\n(Lower α → WNO dominant; Higher α → FNO dominant)', 
                 fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    gate_plot_path = os.path.join(results_dir, 'awfno_v2_sod_gate_alpha.png')
    plt.savefig(gate_plot_path, dpi=150)
    print(f"Gate α visualization saved to {gate_plot_path}")
    print(f"α range: [{alpha_mean.min():.4f}, {alpha_mean.max():.4f}]")
    print(f"α mean: {alpha_mean.mean():.4f}")
    plt.close()

    # ─── 9. Save Model ──────────────────────────────────────────────
    model_path = os.path.join(results_dir, 'awfno_v2_sod_best.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save metadata
    metadata = {
        'downsample_factor': downsample_factor,
        'lr_resolution': lr_resolution,
        'hr_resolution': hr_resolution,
        'n_params': n_params,
        'rel_l2_per_field': rel_l2_per_field,
        'epochs_trained': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }
    
    import json
    metadata_path = os.path.join(results_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    train_sod()
