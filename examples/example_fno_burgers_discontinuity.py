"""
Burgers equation with discontinuity in the velocity field — FNO version.

Paper reference: WNO paper (Tripura & Chakraborty, CMA 2023), Section 4.1 case study.

PDE:
    ∂_t u + u ∂_x u = (0.01/π) ∂_xx u,   x ∈ [-1, 1], t ∈ [0, 1]
    u(-1, t) = u(1, t) = 0                  (zero Dirichlet BC — NOT periodic)
    u(x, 0)  = -sin(πx) + ζ sin(πx),       ζ ~ Uniform[0, 0.5]

Operator being learned:
    u|_{[-1,1] × [0, T_in·Δt]}  →  u|_{[-1,1] × (T_in·Δt, (T_in+T_out)·Δt]}
    i.e., given the first T_in=10 time snapshots, predict the next T_out=40.

Spatial resolution: N_x = 512, time step: Δt = 0.02, total steps: N_t = 50.

Data source: MATLAB PDE solver toolbox (generate externally and save as .mat file).

2D-operator strategy:
    The 1D spatial PDE is lifted to a 2D problem by treating (x, t_window) as
    the 2D "spatial" domain. Each forward pass maps one T_in-step window to the
    next T_in-step window:

        model: (B, 1, N_x, T_in) → (B, 1, N_x, T_in)

    Autoregressive rollout at test time: 4 passes cover all T_out=40 future steps.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not found. Only .pt data loading will work.")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# FNO is dimension-agnostic: passing n_modes=(16, 4) (a 2-tuple) automatically
# configures it as a 2D operator, treating the input (B, 1, N_x, T_in) as a
# 2D spatial-temporal field.
# ---------------------------------------------------------------------------
from awfno.models.fno import FNO
from awfno.utils.unit_gaussian_normalization import UnitGaussianNormalizer
from awfno.utils.losses import LpLoss
from awfno.utils.seed import set_seed


# ===========================================================================
# Configuration
# ===========================================================================
T_IN  = 10   # number of input time snapshots (context window)
T_OUT = 40   # number of future steps to predict via autoregressive rollout
N_X   = 512  # spatial resolution (must match dataset)
assert T_OUT % T_IN == 0, "T_OUT must be divisible by T_IN for clean window rollout."
N_ROLLOUT = T_OUT // T_IN  # = 4 autoregressive steps


def load_mat_data(mat_path: str) -> torch.Tensor:
    """
    Load Burgers discontinuity data from a MATLAB .mat file.

    Expected MATLAB variable: 'u_data' of shape (N_samples, N_x, N_t).
    Common alternative key names are tried in order.

    MATLAB stores arrays in column-major order. scipy.io.loadmat transparently
    converts them to row-major (C order) numpy arrays, so the shape you see in
    Python already matches the MATLAB indexing convention.

    If the array has a different axis order in your file, uncomment and adapt
    the transpose line below.
    """
    assert SCIPY_AVAILABLE, "scipy is required to load .mat files."
    raw = loadmat(mat_path)

    for key in ['u_data', 'u', 'usol', 'sol', 'data', 'output']:
        if key in raw and not key.startswith('__'):
            arr = raw[key]
            print(f"  Loaded key '{key}' with shape {arr.shape}")
            break
    else:
        user_keys = [k for k in raw if not k.startswith('__')]
        raise KeyError(f"None of the expected keys found. Available keys: {user_keys}")

    # If your MATLAB file stored the data transposed, uncomment:
    # arr = arr.transpose(2, 1, 0)   # (N_t, N_x, N) → (N, N_x, N_t)

    return torch.tensor(arr, dtype=torch.float32)


def build_windows(data: torch.Tensor, t_in: int) -> tuple:
    """
    Build non-overlapping (input_window, target_window) pairs from full trajectories.

    Args:
        data : (N_samples, N_x, N_t) — full spatiotemporal trajectories.
        t_in : number of time steps per window.

    Returns:
        x : (N_windows, N_x, t_in) — input windows.
        y : (N_windows, N_x, t_in) — immediately following windows (targets).
    """
    N, Nx, Nt = data.shape
    n_pairs = Nt // t_in - 1

    xs, ys = [], []
    for i in range(n_pairs):
        xs.append(data[:, :, i * t_in : (i + 1) * t_in])
        ys.append(data[:, :, (i + 1) * t_in : (i + 2) * t_in])

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return x, y


def train_burgers_discontinuity():
    set_seed(42)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    epochs       = 500
    batch_size   = 20
    lr           = 1e-3
    print_every  = 50
    n_train      = 1000
    n_test       = 100

    data_dir    = '/home/parikshit/AW-FNO/awfno/data/burger'
    mat_file    = os.path.join(data_dir, 'burgers_data_512_51.mat')
    pt_file     = os.path.join(data_dir, 'burgers_disc.pt')
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'fno_burgers_discontinuity')
    os.makedirs(results_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load data — prefer cached .pt, fall back to .mat
    # -----------------------------------------------------------------------
    print("Loading Burgers discontinuity data...")
    if os.path.exists(pt_file):
        data = torch.load(pt_file)
        print(f"  Loaded from cached .pt file, shape: {data.shape}")
    elif os.path.exists(mat_file):
        data = load_mat_data(mat_file)
        torch.save(data, pt_file)
        print(f"  Converted .mat → .pt (shape {data.shape}); cached at {pt_file}")
    else:
        raise FileNotFoundError(
            f"No data file found.\n"
            f"  Expected .pt : {pt_file}\n"
            f"  Expected .mat: {mat_file}\n"
            "Generate the data with the MATLAB PDE solver and save as 'burgers_disc.mat'."
        )

    assert data.ndim == 3, f"Expected 3-D data (N, N_x, N_t), got shape {data.shape}"
    assert data.shape[1] == N_X, f"Expected N_x={N_X}, got {data.shape[1]}"
    assert data.shape[2] >= T_IN + T_OUT

    # Clip n_test/n_train to whatever the dataset actually contains.
    n_total = data.shape[0]
    n_test  = min(n_test,  max(1, n_total // 5))   # at most 20 %, at least 1
    n_train = min(n_train, n_total - n_test)
    assert n_train > 0, f"Not enough samples: n_total={n_total}, n_test={n_test}"
    print(f"  Split: {n_train} train / {n_test} test  (total {n_total})")

    data = data[:, :, : T_IN + T_OUT]
    train_data = data[:n_train]
    test_data  = data[n_train : n_train + n_test]

    # -----------------------------------------------------------------------
    # Build non-overlapping window pairs for training
    # -----------------------------------------------------------------------
    x_train_raw, y_train_raw = build_windows(train_data, T_IN)

    # -----------------------------------------------------------------------
    # Reshape for 2D model: add channel dimension.
    # Broadcasts the 1D problem to 2D by treating time steps as the second
    # spatial axis. FNO2d (configured by passing n_modes as a 2-tuple) then
    # performs spectral convolution over the (N_x, T_IN) = (512, 10) grid.
    # -----------------------------------------------------------------------
    x_train = x_train_raw.unsqueeze(1)  # (4000, 1, 512, 10)  ← channel dim added here
    y_train = y_train_raw.unsqueeze(1)  # (4000, 1, 512, 10)

    x_test  = test_data[:, :, :T_IN].unsqueeze(1)    # (100, 1, 512, 10)
    y_test  = test_data[:, :, T_IN:T_IN + T_OUT]      # (100, 512, 40) — full future

    # -----------------------------------------------------------------------
    # Normalise
    # -----------------------------------------------------------------------
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train_n    = x_normalizer.encode(x_train)
    x_test_n     = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train_n    = y_normalizer.encode(y_train)

    train_loader = DataLoader(
        TensorDataset(x_train_n, y_train_n),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test_n, y_test),
        batch_size=batch_size, shuffle=False
    )

    # -----------------------------------------------------------------------
    # Model — FNO configured as 2D by passing a 2-tuple for n_modes.
    # n_modes = (16, 4): keep 16 Fourier modes in the spatial (x) direction
    #                    and 4 modes in the temporal (t_window) direction.
    # The small temporal mode count (4) is appropriate for T_IN=10.
    # domain_padding: adds a small periodic buffer around the non-periodic
    #                 Dirichlet boundary to reduce spectral boundary artefacts.
    # -----------------------------------------------------------------------
    # Parameter budget guide (per spectral-conv layer):
    #   params = 2 × hidden² × n_modes_x × n_modes_y   (×2 for real+imag)
    #   hidden=32, modes=(16,4)  →  131K/layer →  ~210K total   ← original
    #   hidden=64, modes=(32,4)  →  1.05M/layer → ~4.4M total   ← moderate
    #   hidden=64, modes=(64,4)  →  2.1M/layer  → ~8.8M total   ← large
    # n_modes_x is capped at N_x//2=256; n_modes_y is capped at T_IN//2=5.
    # channel_mlp_expansion=1.0 doubles the MLP hidden size (32→64 here), adding
    # 2 × hidden² = 8K per layer — negligible vs spectral conv.
    model = FNO(
        n_modes=(64, 4),            # ↑ from (16,4); n_modes_y capped at T_IN//2=5
        in_channels=1,
        out_channels=1,
        hidden_channels=64,         # ↑ from 32; main driver of parameter count
        n_layers=4,
        positional_embedding="grid",
        use_channel_mlp=True,
        channel_mlp_expansion=1.0,  # ↑ from 0.5; MLP hidden = hidden_channels
        fno_skip="linear",
        domain_padding=0.05,        # small padding to handle Dirichlet (non-periodic) BCs
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    criterion_rel = LpLoss(d=2, p=2, size_average=True)
    criterion_mse = nn.MSELoss()

    x_normalizer.to(device)
    y_normalizer.to(device)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    train_mse_hist, train_rel_hist = [], []
    test_mse_hist,  test_rel_hist  = [], []

    def _fmt_time(seconds: float) -> str:
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s   = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"

    print(f"Starting FNO training for {epochs} epochs...")
    print(f"{'Epoch':>6} | {'Tr MSE':>10} | {'Tr RelL2':>9} | "
          f"{'Te MSE':>10} | {'Te RelL2':>9} | {'t_ep':>7} | {'ETA':>10}")
    print("-" * 78)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        t_ep = time.time()

        # ---- Training ----
        model.train()
        train_mse = train_rel = 0.0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()

            pred     = model(bx)
            pred_dec = y_normalizer.decode(pred)
            by_dec   = y_normalizer.decode(by)

            # MSE drives training for FNO (no H1 loss here).
            loss = criterion_mse(pred_dec, by_dec)
            loss.backward()
            optimizer.step()

            train_mse += loss.item()
            train_rel += criterion_rel(
                pred_dec.view(pred_dec.size(0), -1),
                by_dec.view(by_dec.size(0), -1)
            ).item()

        n_batches  = len(train_loader)
        train_mse /= n_batches
        train_rel /= n_batches
        train_mse_hist.append(train_mse)
        train_rel_hist.append(train_rel)

        # ---- Validation: 4-step autoregressive rollout ----
        model.eval()
        test_mse = test_rel = 0.0
        with torch.no_grad():
            for bx_n, by_full in test_loader:
                bx_n    = bx_n.to(device)
                by_full = by_full.to(device)

                pred_windows = []
                inp = bx_n
                for _ in range(N_ROLLOUT):
                    out     = model(inp)
                    out_dec = y_normalizer.decode(out)
                    pred_windows.append(out_dec.squeeze(1))
                    # Re-encode prediction as next input window (autoregressive).
                    inp = x_normalizer.encode(out_dec)

                pred_full = torch.cat(pred_windows, dim=-1)   # (B, 512, 40)

                test_mse += criterion_mse(pred_full, by_full).item()
                test_rel += criterion_rel(
                    pred_full.view(pred_full.size(0), -1),
                    by_full.view(by_full.size(0), -1)
                ).item()

        test_mse /= len(test_loader)
        test_rel /= len(test_loader)
        test_mse_hist.append(test_mse)
        test_rel_hist.append(test_rel)

        scheduler.step()

        # ---- Per-epoch timing & ETA ----
        ep_time = time.time() - t_ep
        elapsed = time.time() - t0
        eta     = elapsed / epoch * (epochs - epoch)

        print(f"{epoch:>6}/{epochs} | {train_mse:>10.4e} | {train_rel:>9.4f} | "
              f"{test_mse:>10.4e} | {test_rel:>9.4f} | "
              f"{ep_time:>6.1f}s | {_fmt_time(eta):>10}")

    elapsed_total = time.time() - t0
    print("-" * 78)
    print(f"Training completed in {_fmt_time(elapsed_total)}  |  "
          f"Final Test Rel L2: {test_rel_hist[-1]:.6f}")

    torch.save(model.state_dict(), os.path.join(results_dir, 'fno_burgers_disc_model.pt'))

    # -----------------------------------------------------------------------
    # Plot 1: Training / testing loss curves
    # -----------------------------------------------------------------------
    epochs_ax = range(1, epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].semilogy(epochs_ax, train_mse_hist, label='Train MSE')
    axes[0].semilogy(epochs_ax, test_mse_hist,  label='Test MSE')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE (log scale)')
    axes[0].set_title('FNO Burgers Discontinuity — MSE')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(epochs_ax, train_rel_hist, label='Train Rel L2')
    axes[1].semilogy(epochs_ax, test_rel_hist,  label='Test Rel L2')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Relative L2 (log scale)')
    axes[1].set_title('FNO Burgers Discontinuity — Relative L2')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    loss_plot = os.path.join(results_dir, 'fno_burgers_disc_loss.png')
    plt.savefig(loss_plot, dpi=150)
    plt.close()
    print(f"Loss plot saved: {loss_plot}")

    # -----------------------------------------------------------------------
    # Plot 2 + 3: Ground truth vs prediction vs error
    # -----------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        sample_x_n = x_test_n[0:1].to(device)
        sample_gt  = y_test[0].numpy()            # (512, 40)

        pred_wins = []
        inp = sample_x_n
        for _ in range(N_ROLLOUT):
            out     = model(inp)
            out_dec = y_normalizer.decode(out)
            pred_wins.append(out_dec.squeeze(1).cpu())
            inp = x_normalizer.encode(out_dec)
        sample_pred = torch.cat(pred_wins, dim=-1).squeeze(0).numpy()  # (512, 40)

    x_coords = np.linspace(-1, 1, N_X)
    t_coords = np.arange(T_IN, T_IN + T_OUT) * 0.02

    # Spatiotemporal heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vmin, vmax = sample_gt.min(), sample_gt.max()

    im0 = axes[0].imshow(sample_gt,   origin='lower', aspect='auto',
                          extent=[t_coords[0], t_coords[-1], x_coords[0], x_coords[-1]],
                          vmin=vmin, vmax=vmax, cmap='RdBu_r')
    axes[0].set_title('Ground Truth'); axes[0].set_xlabel('t'); axes[0].set_ylabel('x')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(sample_pred, origin='lower', aspect='auto',
                          extent=[t_coords[0], t_coords[-1], x_coords[0], x_coords[-1]],
                          vmin=vmin, vmax=vmax, cmap='RdBu_r')
    axes[1].set_title('FNO Prediction'); axes[1].set_xlabel('t'); axes[1].set_ylabel('x')
    plt.colorbar(im1, ax=axes[1])

    err = np.abs(sample_gt - sample_pred)
    im2 = axes[2].imshow(err,         origin='lower', aspect='auto',
                          extent=[t_coords[0], t_coords[-1], x_coords[0], x_coords[-1]],
                          cmap='hot_r')
    axes[2].set_title('|Error|'); axes[2].set_xlabel('t'); axes[2].set_ylabel('x')
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(f'FNO Burgers Discontinuity  |  Test Rel L2: {test_rel_hist[-1]:.4f}')
    plt.tight_layout()
    heatmap_plot = os.path.join(results_dir, 'fno_burgers_disc_heatmap.png')
    plt.savefig(heatmap_plot, dpi=150)
    plt.close()
    print(f"Spatiotemporal heatmap saved: {heatmap_plot}")

    # Temporal snapshot comparisons (4 time instants, like Figure 5 in the paper)
    snap_indices = [T_OUT // 4 - 1, T_OUT // 2 - 1, 3 * T_OUT // 4 - 1, T_OUT - 1]
    snap_times   = [(idx + T_IN) * 0.02 for idx in snap_indices]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, idx, t_snap in zip(axes, snap_indices, snap_times):
        ax.plot(x_coords, sample_gt[:, idx],   '-',  color='navy',   lw=2, label='Truth')
        ax.plot(x_coords, sample_pred[:, idx], '--', color='crimson', lw=2, label='FNO')
        ax.set_title(f't = {t_snap:.2f} s')
        ax.set_xlabel('x')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('u(x, t)')
    fig.suptitle('FNO Burgers Discontinuity — Temporal Snapshots')
    plt.tight_layout()
    snap_plot = os.path.join(results_dir, 'fno_burgers_disc_snapshots.png')
    plt.savefig(snap_plot, dpi=150)
    plt.close()
    print(f"Temporal snapshots plot saved: {snap_plot}")


if __name__ == '__main__':
    train_burgers_discontinuity()
