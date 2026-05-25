"""
Shared utilities for SOD shock super-resolution experiments.

All baseline scripts (FNO, WNO, U-Net) and the AW-FNO enhanced script use
identical data loading, train/test splitting, training loop, evaluation, and
plotting via the run_experiment() entry point defined here.

Fair-comparison guarantees:
  - Same data files, same normalisation
  - Same seed (42) → identical train/test split across all scripts
  - Same optimiser: Adam, lr=1e-3, weight_decay=1e-4
  - Same scheduler: StepLR(step_size=100, gamma=0.5)
  - Same loss: MSELoss during training; relative L2 for final reporting
  - Same ×4 SuperResolutionWrapper for every model
"""

import json
import os
import random
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─── Shared hyper-parameters ─────────────────────────────────────────────────
DATA_ROOT       = '/media/HDD/mamta_backup/datasets/PDEBench/comp_ns/1d'
SEED            = 42
EPOCHS          = 500
BATCH_SIZE      = 16
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
LR_STEP         = 100
LR_GAMMA        = 0.5
DOWNSAMPLE_FACTOR = 4
LR_RESOLUTION   = 256    # 1024 // 4
HR_RESOLUTION   = 1024
FIELD_NAMES     = ['Vx', 'density', 'pressure']
PRINT_EVERY     = 50

# ─── Device selection ─────────────────────────────────────────────────────────

def best_device() -> torch.device:
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

# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ─── Normalisation ────────────────────────────────────────────────────────────

def normalize_field(x):
    """Z-score normalisation per field over (samples, spatial) dims.

    Args:
        x: (N, 3, L) tensor

    Returns:
        x_norm, mu (1,3,1), std (1,3,1)
    """
    mu  = x.mean(dim=(0, 2), keepdim=True)
    std = x.std(dim=(0, 2),  keepdim=True)
    return (x - mu) / (std + 1e-8), mu, std


def denormalize_field(x_norm, mu, std):
    if isinstance(x_norm, np.ndarray):
        mu_np  = mu.numpy()  if isinstance(mu,  torch.Tensor) else mu
        std_np = std.numpy() if isinstance(std, torch.Tensor) else std
        return x_norm * std_np + mu_np
    mu_t  = mu  if isinstance(mu,  torch.Tensor) else torch.from_numpy(mu)
    std_t = std if isinstance(std, torch.Tensor) else torch.from_numpy(std)
    return x_norm * std_t + mu_t

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_sod_data(data_root: str = DATA_ROOT, downsample_factor: int = DOWNSAMPLE_FACTOR):
    """Load and normalise Sod1 + Sod3 + Sod5.

    Returns:
        lr_norm  : (N, 3, 256)  low-res normalised
        hr_norm  : (N, 3, 1024) high-res normalised
        mu       : (1, 3, 1)
        std      : (1, 3, 1)
    """
    all_hr = []
    for name in ['1D_CFD_Sod1.hdf5', '1D_CFD_Sod3.hdf5', '1D_CFD_Sod5.hdf5']:
        path = os.path.join(data_root, name)
        print(f"  Loading {name} …")
        with h5py.File(path, 'r') as f:
            chunk = np.stack([f['Vx'][:], f['density'][:], f['pressure'][:]], axis=1)
        all_hr.append(chunk)
        print(f"    shape: {chunk.shape}")

    hr = torch.from_numpy(np.concatenate(all_hr, axis=0)).float()
    print(f"Combined HR shape: {hr.shape}  "
          f"Vx [{hr[:,0].min():.3f}, {hr[:,0].max():.3f}]  "
          f"density [{hr[:,1].min():.3f}, {hr[:,1].max():.3f}]  "
          f"pressure [{hr[:,2].min():.3f}, {hr[:,2].max():.3f}]")

    hr_norm, mu, std = normalize_field(hr)
    lr_norm = hr_norm[:, :, ::downsample_factor]
    print(f"LR shape (×{downsample_factor} downsampled): {lr_norm.shape}")
    return lr_norm, hr_norm, mu, std


def split_data(lr, hr, seed: int = SEED):
    """N-1 train / 1 test split — one unseen sample held out for evaluation.

    The single test sample is drawn randomly but reproducibly via seed.
    All other N-1 samples are used for training.
    """
    rng = np.random.RandomState(seed)
    n = lr.shape[0]
    idx = rng.permutation(n)
    te, tr = idx[:1], idx[1:]          # first shuffled index → test
    return lr[tr], lr[te], hr[tr], hr[te], tr, te

# ─── Upsampling wrapper ───────────────────────────────────────────────────────

class SuperResolutionWrapper(nn.Module):
    """Runs base_model at LR resolution then linearly upsamples to HR."""
    def __init__(self, base_model: nn.Module, hr_size: int = HR_RESOLUTION):
        super().__init__()
        self.base_model = base_model
        self.hr_size = hr_size

    def forward(self, x):
        out = self.base_model(x)
        return nn.functional.interpolate(out, size=self.hr_size,
                                          mode='linear', align_corners=True)

# ─── Training & evaluation ────────────────────────────────────────────────────

def _train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, per_field = 0.0, [0.0, 0.0, 0.0]
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        total += loss.item()
        for fi in range(3):
            per_field[fi] += criterion(out[:, fi, :], by[:, fi, :]).item()
    n = len(loader)
    return total / n, [p / n for p in per_field]


def _eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total, per_field = 0.0, [0.0, 0.0, 0.0]
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            total += criterion(out, by).item()
            for fi in range(3):
                per_field[fi] += criterion(out[:, fi, :], by[:, fi, :]).item()
    n = len(loader)
    return total / n, [p / n for p in per_field]


def run_experiment(
    model_name:  str,
    base_model:  nn.Module,
    results_dir: str,
    data_root:   str  = DATA_ROOT,
    epochs:      int  = EPOCHS,
    batch_size:  int  = BATCH_SIZE,
    lr:          float = LEARNING_RATE,
    extra_meta:  dict  = None,
):
    """Full train + evaluate pipeline for one model.

    Saves to results_dir:
      - <model_name>_training_loss.png
      - <model_name>_shock_profiles.png
      - metadata.json   (including rel_l2_per_field and n_params)
      - <model_name>_best.pt

    Returns:
        rel_l2_per_field : dict {field_name: error}
        n_params         : int
    """
    os.makedirs(results_dir, exist_ok=True)
    set_seed(SEED)

    device = best_device()
    print(f"\n{'='*60}")
    print(f"  {model_name}  |  device: {device}")
    print(f"{'='*60}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading data …")
    data_lr, data_hr, mu, std = load_sod_data(data_root)
    x_train, x_test, y_train, y_test, tr_idx, te_idx = split_data(data_lr, data_hr)
    print(f"Train: {len(tr_idx)}  Test: {len(te_idx)}")

    train_loader = DataLoader(TensorDataset(x_train, y_train),
                               batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(x_test,  y_test),
                               batch_size=batch_size, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = SuperResolutionWrapper(base_model, HR_RESOLUTION).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)
    criterion = nn.MSELoss()

    # ── Training loop ────────────────────────────────────────────────────────
    train_hist, test_hist = [], []
    train_pf = {fi: [] for fi in range(3)}
    test_pf  = {fi: [] for fi in range(3)}

    print(f"Training for {epochs} epochs …")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_pf = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_pf = _eval_one_epoch(model, test_loader, criterion, device)
        scheduler.step()

        train_hist.append(tr_loss)
        test_hist.append(te_loss)
        for fi in range(3):
            train_pf[fi].append(tr_pf[fi])
            test_pf[fi].append(te_pf[fi])

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}/{epochs} | "
                  f"Train {tr_loss:.6f} | Test {te_loss:.6f}")

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s")

    # ── Final evaluation ──────────────────────────────────────────────────────
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            preds.append(model(bx).cpu())
            trues.append(by.cpu())
    pred_all = torch.cat(preds, dim=0)
    true_all = torch.cat(trues, dim=0)

    print(f"\nPER-FIELD RELATIVE L2 ERRORS")
    rel_l2 = {}
    for fi, fname in enumerate(FIELD_NAMES):
        e = (torch.norm(pred_all[:, fi, :] - true_all[:, fi, :]) /
             torch.norm(true_all[:, fi, :])).item()
        rel_l2[fname] = e
        print(f"  {fname:12s}: {e:.6f}")

    # ── Plot: training loss ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} — SOD Super-Resolution', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(train_hist, label='Train', linewidth=2)
    ax.plot(test_hist,  label='Test',  linewidth=2)
    ax.set(xlabel='Epoch', ylabel='MSE Loss', title='Overall Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for fi, fname in enumerate(FIELD_NAMES):
        ax.plot(train_pf[fi], label=fname, linewidth=2)
    ax.set(xlabel='Epoch', ylabel='MSE Loss', title='Per-Field Train Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for fi, fname in enumerate(FIELD_NAMES):
        ax.plot(test_pf[fi], label=fname, linewidth=2)
    ax.set(xlabel='Epoch', ylabel='MSE Loss', title='Per-Field Test Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ratio = np.array(test_hist) / (np.array(train_hist) + 1e-8)
    ax.plot(ratio, color='red', linewidth=2, label='Test/Train')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set(xlabel='Epoch', ylabel='Ratio', title='Overfitting Indicator')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name}_training_loss.png'), dpi=150)
    plt.close()

    # ── Plot: shock profiles ──────────────────────────────────────────────────
    n_viz = min(4, len(te_idx))
    rng = np.random.RandomState(SEED)
    sample_indices = rng.choice(len(te_idx), n_viz, replace=False)

    fig, axes = plt.subplots(n_viz, 3, figsize=(15, 4 * n_viz))
    if n_viz == 1:
        axes = axes.reshape(1, -1)

    x_hr = np.arange(HR_RESOLUTION)
    for sn, si in enumerate(sample_indices):
        p = denormalize_field(pred_all[si].numpy().T.reshape(1, 3, -1),
                              mu.reshape(1, 3, 1), std.reshape(1, 3, 1)).squeeze()
        t = denormalize_field(true_all[si].numpy().T.reshape(1, 3, -1),
                              mu.reshape(1, 3, 1), std.reshape(1, 3, 1)).squeeze()
        for fi, fname in enumerate(FIELD_NAMES):
            ax = axes[sn, fi]
            ax.plot(x_hr, t[fi], 'b-',  linewidth=2.5, label='Ground Truth', alpha=0.8)
            ax.plot(x_hr, p[fi], 'r--', linewidth=2.0, label=model_name,    alpha=0.8)
            ax.set(xlabel='Spatial Position', ylabel=fname,
                   title=f'Sample {si+1} — {fname}')
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name}_shock_profiles.png'), dpi=150)
    plt.close()

    # ── Save model & metadata ─────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(results_dir, f'{model_name}_best.pt'))

    meta = {
        'model_name':        model_name,
        'n_params':          n_params,
        'rel_l2_per_field':  rel_l2,
        'rel_l2_mean':       float(np.mean(list(rel_l2.values()))),
        'epochs_trained':    epochs,
        'batch_size':        batch_size,
        'learning_rate':     lr,
        'downsample_factor': DOWNSAMPLE_FACTOR,
        'lr_resolution':     LR_RESOLUTION,
        'hr_resolution':     HR_RESOLUTION,
        'training_time_s':   round(elapsed, 1),
    }
    if extra_meta:
        meta.update(extra_meta)

    with open(os.path.join(results_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nResults saved to {results_dir}")
    return rel_l2, n_params
