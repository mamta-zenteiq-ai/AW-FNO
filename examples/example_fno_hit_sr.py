"""
FNO — Homogeneous Isotropic Turbulence (HIT) super-resolution.

Setup
-----
Source data: /home/parikshit/data_HIT/high_res/{train,val}.npy   shape: (N, 128, 128)
LR generation: avg_pool2d on HR (no separate LR file needed)
  - Training : LR = avg_pool(HR, kernel=4) → 32x32, then bicubic-up to 128
  - Test     : same recipe (32 → 128) on unseen val samples

Model: FNO operates at 128x128 throughout (pre-upsample LR → 128). Bicubic
pre-upsampling here mirrors the AW-FNO script so the two are directly
comparable — FNO alone is mode-based and could in principle accept raw 32x32
input, but to keep an apples-to-apples comparison with AW-FNO we use the
same pipeline.

Loss : MSE for training and reporting.
Plots: 4-panel (GT | bicubic-upsampled LR | FNO prediction | |error|) + loss curves.
Out  : results/fno_hit_sr/
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from awfno.models.fno import FNO
from awfno.utils.unit_gaussian_normalization import UnitGaussianNormalizer
from awfno.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_lr_then_upsample(hr, lr_size, hr_size):
    """Downsample HR to LR via avg_pool, then bicubic-up to hr_size."""
    kernel = hr.shape[-1] // lr_size
    lr = F.avg_pool2d(hr, kernel_size=kernel)
    return F.interpolate(lr, size=hr_size, mode='bicubic', align_corners=False)


def relative_l2(pred, true, eps=1e-8):
    n = pred.shape[0]
    return (torch.norm((pred - true).reshape(n, -1), dim=1) /
            (torch.norm(true.reshape(n, -1), dim=1) + eps)).mean().item()


def relative_l2_sum(pred, true, eps=1e-8):
    """Sum over batch (not mean) — for accumulating across an epoch."""
    n = pred.shape[0]
    return (torch.norm((pred - true).reshape(n, -1), dim=1) /
            (torch.norm(true.reshape(n, -1), dim=1) + eps)).sum().item()


def psnr(gt, pr):
    """Peak Signal-to-Noise Ratio (dB) for a single 2D field."""
    mse = float(np.mean((gt - pr) ** 2))
    if mse == 0.0:
        return float('inf')
    data_range = float(gt.max() - gt.min())
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


def ssim(gt, pr, sigma=1.5, k1=0.01, k2=0.03):
    """Structural Similarity Index (Wang et al. 2004) with a Gaussian window."""
    from scipy.ndimage import gaussian_filter
    gt = gt.astype(np.float64)
    pr = pr.astype(np.float64)
    data_range = float(gt.max() - gt.min())
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    mu_x = gaussian_filter(gt,      sigma)
    mu_y = gaussian_filter(pr,      sigma)
    s_xx = gaussian_filter(gt * gt, sigma) - mu_x * mu_x
    s_yy = gaussian_filter(pr * pr, sigma) - mu_y * mu_y
    s_xy = gaussian_filter(gt * pr, sigma) - mu_x * mu_y
    num  = (2 * mu_x * mu_y + c1) * (2 * s_xy + c2)
    den  = (mu_x ** 2 + mu_y ** 2 + c1) * (s_xx + s_yy + c2)
    return float(np.mean(num / den))


def fmt_time(s):
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:   return f"{h}h{m:02d}m{sec:02d}s"
    if m:   return f"{m}m{sec:02d}s"
    return f"{sec}s"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fno_hit_sr():
    set_seed(42)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Config ---
    epochs        = 500
    batch_size    = 16
    learning_rate = 1e-3
    print_every   = 10

    lr_train = 32
    hr_size  = 128

    data_path   = '/home/parikshit/data_HIT/high_res'
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'fno_hit_sr')
    os.makedirs(results_dir, exist_ok=True)

    # --- Load HR data ---
    print(f"Loading HR HIT data from {data_path} ...")
    y_train_hr = torch.from_numpy(np.load(os.path.join(data_path, 'train.npy'))).float().unsqueeze(1)
    y_test_hr  = torch.from_numpy(np.load(os.path.join(data_path, 'val.npy'))).float().unsqueeze(1)
    print(f"  y_train_hr: {tuple(y_train_hr.shape)}  y_test_hr: {tuple(y_test_hr.shape)}")

    # --- Build LR inputs by downsample-then-upsample ---
    x_train        = make_lr_then_upsample(y_train_hr, lr_train, hr_size)  # 32 → 128
    x_test_in_phys = make_lr_then_upsample(y_test_hr,  lr_train, hr_size)  # 32 → 128

    # --- Normalize ---
    x_norm = UnitGaussianNormalizer(x_train)
    x_train   = x_norm.encode(x_train)
    x_test_in = x_norm.encode(x_test_in_phys)

    y_norm = UnitGaussianNormalizer(y_train_hr)
    y_train_norm = y_norm.encode(y_train_hr)
    y_norm.to(device)

    train_loader   = DataLoader(TensorDataset(x_train,   y_train_norm),
                                batch_size=batch_size, shuffle=True)
    test_loader_in = DataLoader(TensorDataset(x_test_in, y_test_hr),
                                batch_size=batch_size, shuffle=False)

    # --- Model ---
    model = FNO(
        n_modes=(20, 20),
        in_channels=1,
        out_channels=1,
        hidden_channels=192,
        n_layers=4,
        positional_embedding="grid",
        use_channel_mlp=True,
        channel_mlp_dropout=0.0,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    mse_loss  = nn.MSELoss()

    # --- Train ---
    train_mse_hist   = []
    test_mse_in_hist = []
    train_rl2_hist   = []
    test_rl2_in_hist = []

    print(f"Training FNO on HIT SR ({lr_train}→{hr_size}) for {epochs} epochs ...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        running_mse = 0.0
        running_rl2 = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out  = model(bx)
            loss = mse_loss(out, by)
            loss.backward()
            optimizer.step()
            running_mse += loss.item() * bx.size(0)
            with torch.no_grad():
                running_rl2 += relative_l2_sum(y_norm.decode(out), y_norm.decode(by))
        n_tr = len(train_loader.dataset)
        train_mse_hist.append(running_mse / n_tr)
        train_rl2_hist.append(running_rl2 / n_tr)

        model.eval()
        with torch.no_grad():
            in_sse, in_rl2, n_in = 0.0, 0.0, 0
            for bx, by in test_loader_in:
                bx, by = bx.to(device), by.to(device)
                out = y_norm.decode(model(bx))
                in_sse += ((out - by) ** 2).mean(dim=(1, 2, 3)).sum().item()
                in_rl2 += relative_l2_sum(out, by)
                n_in   += bx.size(0)
        test_mse_in_hist.append(in_sse / n_in)
        test_rl2_in_hist.append(in_rl2 / n_in)

        scheduler.step()

        ep_time = time.time() - t_ep
        eta     = ep_time * (epochs - epoch)
        print(f"Epoch {epoch:4d}/{epochs} | "
              f"train MSE: {train_mse_hist[-1]:.4e}  RelL2: {train_rl2_hist[-1]:.4f} | "
              f"test MSE: {test_mse_in_hist[-1]:.4e}  RelL2: {test_rl2_in_hist[-1]:.4f} | "
              f"epoch: {ep_time:5.1f}s | ETA: {fmt_time(eta)}")

    total_time = time.time() - t0
    print(f"Training done in {fmt_time(total_time)} ({total_time:.1f}s, "
          f"avg {total_time/epochs:.1f}s/epoch)")

    # --- Final relative-L2 report (physical scale, on both train + test) ---
    model.eval()
    train_eval_loader = DataLoader(TensorDataset(x_train, y_train_hr),
                                   batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        all_tr, all_tr_y = [], []
        for bx, by in train_eval_loader:
            all_tr  .append(y_norm.decode(model(bx.to(device))))
            all_tr_y.append(by.to(device))
        pred_tr  = torch.cat(all_tr)
        y_tr_all = torch.cat(all_tr_y)

        all_in, all_y = [], []
        for bx_in, by in test_loader_in:
            all_in.append(y_norm.decode(model(bx_in.to(device))))
            all_y .append(by.to(device))
        pred_in = torch.cat(all_in)
        y_all   = torch.cat(all_y)

        rl2_train = relative_l2(pred_tr, y_tr_all)
        rl2_test  = relative_l2(pred_in, y_all)
        print(f"Final Rel-L2 train (32→128): {rl2_train:.4f}")
        print(f"Final Rel-L2 test  (32→128): {rl2_test :.4f}")

    # --- Plots ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size']   = 13

    # 1) Loss curves
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ep = np.arange(1, epochs + 1)
    ax.semilogy(ep, train_mse_hist,   'b-', label='Train MSE (32→128)')
    ax.semilogy(ep, test_mse_in_hist, 'r-', label='Test MSE (32→128)')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('MSE',   fontweight='bold')
    ax.set_title('FNO HIT SR — Learning Curves', fontweight='bold')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'loss_curves.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(results_dir, 'loss_curves.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)

    # 2) 5-panel: GT | Raw LR (32×32) | Bicubic-upsampled LR | FNO Prediction | |Error|
    # - Raw LR shown with `interpolation='nearest'` so the 32×32 pixelation is visible
    #   (no implicit smoothing). This is what makes the super-resolution visually evident.
    # - PSNR / SSIM are computed per sample (prediction vs GT) in physical scale.
    n_show = 4
    idx    = np.linspace(0, pred_in.shape[0] - 1, n_show, dtype=int)
    col_titles = [
        f'Ground Truth\n(HR {hr_size}×{hr_size})',
        f'Low-Resolution Input\n(LR {lr_train}×{lr_train})',
        f'Bicubic Pre-upsampled\n(LR {lr_train}×{lr_train} → {hr_size}×{hr_size})',
        f'FNO Prediction\n({hr_size}×{hr_size})',
        f'Absolute Error\n({hr_size}×{hr_size})',
    ]
    cmaps  = ['nipy_spectral', 'nipy_spectral', 'nipy_spectral', 'nipy_spectral', 'jet']
    interp = ['Gaussian',      'nearest',       'Gaussian',      'Gaussian',      'Gaussian']
    fig, axes = plt.subplots(n_show, 5, figsize=(21, 4 * n_show))
    fig.suptitle(f'FNO HIT SR ({lr_train}→{hr_size}) — GT | LR | Bicubic LR | Prediction | |Error|',
                 fontsize=15, fontweight='bold')

    sample_psnr, sample_ssim = [], []
    print("\nPer-sample PSNR / SSIM (FNO prediction vs HR ground truth):")
    for r, i in enumerate(idx):
        gt    = y_all          [i].cpu().numpy().squeeze()
        bc    = x_test_in_phys [i].cpu().numpy().squeeze()
        pr    = pred_in        [i].cpu().numpy().squeeze()
        err   = np.abs(gt - pr)
        lr_gt = F.avg_pool2d(y_all[i:i+1], kernel_size=hr_size // lr_train).cpu().numpy().squeeze()

        ps = psnr(gt, pr)
        ss = ssim(gt, pr)
        sample_psnr.append(ps)
        sample_ssim.append(ss)
        print(f"  Sample {i:3d}: PSNR = {ps:6.2f} dB | SSIM = {ss:.4f}")

        fields = [gt, lr_gt, bc, pr, err]
        for c, (field, cmap, ctitle, ip) in enumerate(zip(fields, cmaps, col_titles, interp)):
            ax = axes[r, c]
            im = ax.imshow(field, origin='lower', extent=[0, 1, 0, 1],
                           interpolation=ip, cmap=cmap)
            plt.colorbar(im, ax=ax, fraction=0.045)
            if r == 0:
                ax.set_title(ctitle, fontsize=12, fontweight='bold')
            ax.set_xlabel('x', fontweight='bold')

        axes[r, 0].set_ylabel(
            f'Sample {i}\nPSNR = {ps:.2f} dB\nSSIM = {ss:.4f}\ny',
            fontweight='bold')

    psnr_mean = float(np.mean(sample_psnr))
    ssim_mean = float(np.mean(sample_ssim))
    print(f"  Mean over {n_show} shown samples: PSNR = {psnr_mean:.2f} dB | SSIM = {ssim_mean:.4f}\n")

    fig.text(0.5, 0.005,
             f'Mean over shown samples — PSNR: {psnr_mean:.2f} dB    SSIM: {ssim_mean:.4f}',
             ha='center', fontsize=13, fontweight='bold')

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(os.path.join(results_dir, 'gt_lr_bicubic_pred_error.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(results_dir, 'gt_lr_bicubic_pred_error.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)

    # 3) Save histories + final preds for reuse
    np.savez(os.path.join(results_dir, 'history.npz'),
             train_mse=np.array(train_mse_hist),
             test_mse_in=np.array(test_mse_in_hist),
             train_rl2=np.array(train_rl2_hist),
             test_rl2_in=np.array(test_rl2_in_hist))
    torch.save(model.state_dict(), os.path.join(results_dir, 'fno_hit_sr.pt'))

    print(f"All results saved to {results_dir}/")


if __name__ == "__main__":
    train_fno_hit_sr()
