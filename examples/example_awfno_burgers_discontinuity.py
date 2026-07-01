"""
Burgers equation with discontinuity in the velocity field — AW-FNO version.

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

# ---------------------------------------------------------------------------
# scipy is required only for loading the MATLAB .mat file.
# If your data is already stored as a .pt file, remove this import and adapt
# the loading section below.
# ---------------------------------------------------------------------------
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
# [COMMENTED OUT] parallel AWFNO2d (jit.fork parallelism + AGFM at every block)
# ---------------------------------------------------------------------------
# from awfno.models.awfno_parallel import AWFNO2d

# ---------------------------------------------------------------------------
# Use AWFNO2dFinalAGFM: FNO and WNO branches run independently for all layers,
# and AdaptiveGatedFusion is applied only ONCE at the very end (vs. awfno_parallel
# which fuses at every block).
# ---------------------------------------------------------------------------
from awfno.models.awfno_finalagfm import AWFNO2dFinalAGFM
from awfno.utils.unit_gaussian_normalization import UnitGaussianNormalizer
from awfno.utils.losses import LpLoss
from awfno.utils.seed import set_seed


# ===========================================================================
# Sobolev (H1) loss — weighted sum of relative L2 value loss and relative L2
# finite-difference gradient loss.  Extended to 2D spatial-temporal fields
# by computing finite differences along both the spatial (x) and temporal (t)
# axes of the (B, C, N_x, T) tensor.
# ===========================================================================
class SobolevLoss2d:
    """
    H1 Sobolev loss for 2D fields of shape (B, C, H, W).

    loss = mean_over_batch( rel_L2(u, u_gt)
                          + beta * rel_L2(∂_x u, ∂_x u_gt)
                          + beta * rel_L2(∂_t u, ∂_t u_gt) )

    Finite differences are computed along the last two axes (H → spatial,
    W → temporal for the Burgers window representation).
    """

    def __init__(self, p: int = 2, beta: float = 1.0, eps: float = 1e-8):
        self.p    = p
        self.beta = beta
        self.eps  = eps

    def _rel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Relative Lp norm, averaged over the batch."""
        a_flat = a.reshape(a.size(0), -1)
        b_flat = b.reshape(b.size(0), -1)
        diff   = torch.norm(a_flat - b_flat, self.p, dim=1)
        norm_b = torch.norm(b_flat,          self.p, dim=1)
        return torch.mean(diff / (norm_b + self.eps))

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Value loss
        loss = self._rel(pred, target)

        # Spatial gradient loss (finite difference along axis -2, i.e. x-axis)
        dx_pred   = pred[..., 1:, :]   - pred[..., :-1, :]    # (B, C, H-1, W)
        dx_target = target[..., 1:, :] - target[..., :-1, :]
        loss = loss + self.beta * self._rel(dx_pred, dx_target)

        # Temporal gradient loss (finite difference along axis -1, i.e. t-axis)
        dt_pred   = pred[..., 1:]   - pred[..., :-1]           # (B, C, H, W-1)
        dt_target = target[..., 1:] - target[..., :-1]
        loss = loss + self.beta * self._rel(dt_pred, dt_target)

        return loss


# ===========================================================================
# Configuration
# ===========================================================================
T_IN  = 10   # number of input time snapshots (context window)
T_OUT = 40   # number of future steps to predict via autoregressive rollout
N_X   = 512  # spatial resolution (must match dataset)
# T_OUT must be divisible by T_IN for the clean non-overlapping window strategy.
# With T_IN=10 and T_OUT=40: exactly 4 rollout steps at test time.
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

    # Try common key names used by MATLAB PDE solver export scripts.
    for key in ['u_data', 'u', 'usol', 'sol', 'data', 'output']:
        if key in raw and not key.startswith('__'):
            arr = raw[key]
            print(f"  Loaded key '{key}' with shape {arr.shape}")
            break
    else:
        # List available user keys to help the user identify the correct one.
        user_keys = [k for k in raw if not k.startswith('__')]
        raise KeyError(f"None of the expected keys found. Available keys: {user_keys}")

    # Expected shape after loading: (N_samples, N_x, N_t).
    # If your MATLAB file stored the data transposed (e.g., shape is (N_t, N_x, N_samples)),
    # uncomment the appropriate transpose:
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

    Each sample contributes N_t//t_in - 1 non-overlapping pairs, which
    increases the effective training set size by that factor.
    """
    N, Nx, Nt = data.shape
    n_pairs = Nt // t_in - 1  # number of window pairs per sample (e.g., 4 for Nt=50, t_in=10)

    xs, ys = [], []
    for i in range(n_pairs):
        # Input window: steps [i*t_in .. (i+1)*t_in - 1]
        xs.append(data[:, :, i * t_in : (i + 1) * t_in])
        # Target window: steps [(i+1)*t_in .. (i+2)*t_in - 1]
        ys.append(data[:, :, (i + 1) * t_in : (i + 2) * t_in])

    # Stack along the sample axis → (N * n_pairs, N_x, t_in)
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return x, y


def train_burgers_discontinuity():
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # Hyperparameters — match WNO paper where applicable
    # -----------------------------------------------------------------------
    epochs       = 500
    batch_size   = 20
    lr           = 1e-3
    print_every  = 50
    n_train      = 1000   # clipped below to actual data size
    n_test       = 100    # clipped below to actual data size

    # -----------------------------------------------------------------------
    # Data paths
    # -----------------------------------------------------------------------
    # The .mat file is expected to contain a variable of shape (N_samples, N_x, N_t).
    # Change the filename to match your MATLAB export.
    data_dir   = '/home/parikshit/AW-FNO/awfno/data/burger'
    mat_file   = os.path.join(data_dir, 'burgers_data_512_51.mat')  # original MATLAB export (slow to load)
    pt_file    = os.path.join(data_dir, 'burgers_disc.pt')   # cached tensor (faster re-runs)
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'awfno_burgers_discontinuity')
    os.makedirs(results_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load data — prefer cached .pt, fall back to .mat
    # -----------------------------------------------------------------------
    print("Loading Burgers discontinuity data...")
    if os.path.exists(pt_file):
        # Cached PyTorch tensor — fast reload after first run.
        data = torch.load(pt_file)
        print(f"  Loaded from cached .pt file, shape: {data.shape}")
    elif os.path.exists(mat_file):
        # First run: load from MATLAB file and cache for subsequent runs.
        data = load_mat_data(mat_file)
        torch.save(data, pt_file)   # cache for faster future loads
        print(f"  Converted .mat → .pt (shape {data.shape}); cached at {pt_file}")
    else:
        raise FileNotFoundError(
            f"No data file found.\n"
            f"  Expected .pt : {pt_file}\n"
            f"  Expected .mat: {mat_file}\n"
            "Generate the data with the MATLAB PDE solver and save as 'burgers_disc.mat'."
        )

    # data shape: (N_samples, N_x, N_t)  e.g. (500, 512, 51)
    assert data.ndim == 3, f"Expected 3-D data (N, N_x, N_t), got shape {data.shape}"
    assert data.shape[1] == N_X, f"Expected N_x={N_X}, got {data.shape[1]}"
    assert data.shape[2] >= T_IN + T_OUT, (
        f"N_t={data.shape[2]} too small; need at least T_IN+T_OUT={T_IN+T_OUT}")

    # Clip n_test/n_train to whatever the dataset actually contains.
    n_total  = data.shape[0]
    n_test   = min(n_test,  max(1, n_total // 5))      # at most 20 % for test, at least 1
    n_train  = min(n_train, n_total - n_test)           # remainder for train
    assert n_train > 0, f"Not enough samples: n_total={n_total}, n_test={n_test}"
    print(f"  Split: {n_train} train / {n_test} test  (total {n_total})")

    # Keep only the time steps we need (T_IN context + T_OUT future = 50)
    data = data[:, :, : T_IN + T_OUT]   # (N, N_x, 50)

    train_data = data[:n_train]
    test_data  = data[n_train : n_train + n_test]

    # -----------------------------------------------------------------------
    # Build non-overlapping input/target window pairs for training.
    # Each trajectory produces N_ROLLOUT=4 pairs → 4000 training pairs total.
    # -----------------------------------------------------------------------
    x_train_raw, y_train_raw = build_windows(train_data, T_IN)
    # x_train_raw: (n_train * n_pairs, 512, 10); n_pairs = T_OUT//T_IN = 4

    # -----------------------------------------------------------------------
    # Reshape for 2D model: add channel dimension.
    # The 2D "spatial" grid is (N_x, T_IN) = (512, 10).
    # This broadcasts the 1D spatial problem into a 2D operator by treating
    # the time-window axis as the second spatial dimension.
    # -----------------------------------------------------------------------
    x_train = x_train_raw.unsqueeze(1)  # (4000, 1, 512, 10)  ← channel dim added here
    y_train = y_train_raw.unsqueeze(1)  # (4000, 1, 512, 10)

    # Test: use only the very first window as input (t=0..9 → predict t=10..49 via rollout)
    x_test = test_data[:, :, :T_IN].unsqueeze(1)     # (100, 1, 512, 10)
    y_test = test_data[:, :, T_IN:T_IN + T_OUT]       # (100, 512, 40)  full future (no channel dim, for plotting)

    # -----------------------------------------------------------------------
    # Normalise (UnitGaussianNormalizer — same as all other scripts)
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
    # [COMMENTED OUT] AWFNO2d — parallel FNO+WNO with AGFM at every block
    # -----------------------------------------------------------------------
    # model = AWFNO2d(
    #     in_channels=1,
    #     out_channels=1,
    #     n_modes=(16, 4),
    #     size=(N_X, T_IN),
    #     hidden_channels=64,
    #     n_layers=4,
    #     wno_level=1,
    #     wno_wavelet='db6',
    #     positional_embedding="grid",
    #     padding=0,
    #     dropout=0.0,
    # ).to(device)

    # -----------------------------------------------------------------------
    # Model — AWFNO2dFinalAGFM (FNO+WNO branches independent for all layers,
    # AdaptiveGatedFusion applied once at the end — lighter fusion overhead)
    # n_modes  : Fourier modes kept in (x, t_window) → (32 spatial, 4 temporal)
    # size     : (N_x, T_IN) = (512, 10) — the 2D grid dimensions
    # wno_level: DWT decomposition level. T_IN=10 limits level to 1
    #            (level 2 would halve 10 → 5 → 2.5, which is not integer).
    # c_gated=1: spatial gating (single gate map) — lightweight fusion.
    # -----------------------------------------------------------------------
    # Parameter budget — WNO wavelet weights dominate and scale as hidden²:
    #   hidden=64 → ~4.3 crore (42.9M)   [original]
    #   hidden=32 → ~1.1 crore (10.7M)
    #   hidden=16 → ~27 lakhs  (2.69M)   ← target ~30 lakhs
    # Formula: params ≈ 42.9M × (hidden/64)²
    # Reducing hidden is the only effective lever because WNO weight shape is
    # (hidden, hidden, N_x/2, T_IN/2) = (hidden, hidden, 256, 5), fixed by grid.
    model = AWFNO2dFinalAGFM(
        in_channels=1,
        out_channels=1,
        n_modes=(24, 4),        # FNO branch modes; capped at N_x//2=256, T_IN//2=5
        size=(N_X, T_IN),       # 2D grid: (512, 10)
        hidden_channels=16,     # ↓ from 32/64; WNO weight_approx=(16,16,256,5)=327K/layer
        n_layers=4,
        wno_level=1,            # level=1 required: T_IN=10 cannot support level≥2 safely
        wno_wavelet='db6',
        positional_embedding="grid",
        padding=0,
        dropout=0.0,
        c_gated=1,              # spatial gating (1 gate map, not per-channel)
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Same decay schedule as other experiments (halve LR every 100 epochs at rate 0.5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    criterion_rel = LpLoss(d=2, p=2, size_average=True)
    criterion_mse = nn.MSELoss()
    # Sobolev H1 loss: value loss + beta*(spatial gradient loss + temporal gradient loss).
    # beta=0.1 keeps the derivative terms as a regulariser without dominating the value loss.
    criterion_h1  = SobolevLoss2d(beta=0.1)

    x_normalizer.to(device)
    y_normalizer.to(device)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    train_h1_hist,  train_mse_hist, train_rel_hist = [], [], []
    test_mse_hist,  test_rel_hist                  = [], []

    def _fmt_time(seconds: float) -> str:
        """Format seconds as h mm ss or mm ss."""
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s   = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"

    print(f"Starting AWFNO training for {epochs} epochs...")
    print(f"{'Epoch':>6} | {'H1 Loss':>10} | {'Tr MSE':>10} | {'Tr RelL2':>9} | "
          f"{'Te MSE':>10} | {'Te RelL2':>9} | {'t_ep':>7} | {'ETA':>10}")
    print("-" * 95)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        t_ep = time.time()

        # ---- Training ----
        model.train()
        train_h1 = train_mse = train_rel = 0.0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()

            pred = model(bx)                        # (B, 1, 512, 10)

            # Decode to physical units before all loss computations.
            pred_dec = y_normalizer.decode(pred)
            by_dec   = y_normalizer.decode(by)

            # Sobolev H1 loss drives training (value + spatial + temporal gradients).
            loss = criterion_h1(pred_dec, by_dec)
            loss.backward()
            optimizer.step()

            train_h1  += loss.item()
            # MSE and Rel L2 tracked separately for reporting — NOT used for backprop.
            train_mse += criterion_mse(pred_dec, by_dec).item()
            train_rel += criterion_rel(
                pred_dec.view(pred_dec.size(0), -1),
                by_dec.view(by_dec.size(0), -1)
            ).item()

        n_batches = len(train_loader)
        train_h1  /= n_batches
        train_mse /= n_batches
        train_rel /= n_batches
        train_h1_hist.append(train_h1)
        train_mse_hist.append(train_mse)
        train_rel_hist.append(train_rel)

        # ---- Validation: autoregressive 4-step rollout ----
        t_test = time.time()
        model.eval()
        test_mse = test_rel = 0.0
        with torch.no_grad():
            for bx_n, by_full in test_loader:
                # bx_n  : (B, 1, 512, 10) — normalised first input window
                # by_full: (B, 512, 40)   — full 40-step ground truth (physical units)
                bx_n    = bx_n.to(device)
                by_full = by_full.to(device)

                pred_windows = []
                inp = bx_n
                for _ in range(N_ROLLOUT):
                    out     = model(inp)
                    out_dec = y_normalizer.decode(out)
                    pred_windows.append(out_dec.squeeze(1))   # (B, 512, 10)
                    # Re-encode: model's own output becomes the next input window.
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
        ep_time  = time.time() - t_ep
        elapsed  = time.time() - t0
        eta      = elapsed / epoch * (epochs - epoch)

        print(f"{epoch:>6}/{epochs} | {train_h1:>10.4e} | {train_mse:>10.4e} | "
              f"{train_rel:>9.4f} | {test_mse:>10.4e} | {test_rel:>9.4f} | "
              f"{ep_time:>6.1f}s | {_fmt_time(eta):>10}")

    elapsed_total = time.time() - t0
    print("-" * 95)
    print(f"Training completed in {_fmt_time(elapsed_total)}  |  "
          f"Final Test Rel L2: {test_rel_hist[-1]:.6f}  |  "
          f"Final Train H1: {train_h1_hist[-1]:.6e}")

    # -----------------------------------------------------------------------
    # Save model
    # -----------------------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(results_dir, 'awfno_burgers_disc_model.pt'))

    # -----------------------------------------------------------------------
    # Plot 1: Training / testing loss curves
    # -----------------------------------------------------------------------
    epochs_ax = range(1, epochs + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].semilogy(epochs_ax, train_h1_hist, label='Train H1 (Sobolev)', color='purple')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('H1 Loss (log scale)')
    axes[0].set_title('AW-FNO Burgers — Sobolev H1 Training Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(epochs_ax, train_mse_hist, label='Train MSE')
    axes[1].semilogy(epochs_ax, test_mse_hist,  label='Test MSE')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MSE (log scale)')
    axes[1].set_title('AW-FNO Burgers Discontinuity — MSE')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(epochs_ax, train_rel_hist, label='Train Rel L2')
    axes[2].semilogy(epochs_ax, test_rel_hist,  label='Test Rel L2')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Relative L2 (log scale)')
    axes[2].set_title('AW-FNO Burgers Discontinuity — Relative L2')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    loss_plot = os.path.join(results_dir, 'awfno_burgers_disc_loss.png')
    plt.savefig(loss_plot, dpi=150)
    plt.close()
    print(f"Loss plot saved: {loss_plot}")

    # -----------------------------------------------------------------------
    # Plot 2 + 3: Ground truth vs prediction vs error for one test sample
    # -----------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        # Take the first test sample for visualisation.
        sample_x_n  = x_test_n[0:1].to(device)      # (1, 1, 512, 10) — normalised input
        sample_gt   = y_test[0].numpy()              # (512, 40) — full ground truth

        pred_wins = []
        inp = sample_x_n
        for _ in range(N_ROLLOUT):
            out      = model(inp)
            out_dec  = y_normalizer.decode(out)
            pred_wins.append(out_dec.squeeze(1).cpu())     # (1, 512, 10)
            inp = x_normalizer.encode(out_dec)
        sample_pred = torch.cat(pred_wins, dim=-1).squeeze(0).numpy()  # (512, 40)

    # --- Plot 2: spatiotemporal heatmaps (ground truth | prediction | error) ---
    x_coords = np.linspace(-1, 1, N_X)
    t_coords = np.arange(T_IN, T_IN + T_OUT) * 0.02   # physical time axis

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
    axes[1].set_title('AW-FNO Prediction'); axes[1].set_xlabel('t'); axes[1].set_ylabel('x')
    plt.colorbar(im1, ax=axes[1])

    err = np.abs(sample_gt - sample_pred)
    im2 = axes[2].imshow(err,         origin='lower', aspect='auto',
                          extent=[t_coords[0], t_coords[-1], x_coords[0], x_coords[-1]],
                          cmap='hot_r')
    axes[2].set_title('|Error|'); axes[2].set_xlabel('t'); axes[2].set_ylabel('x')
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(f'AW-FNO Burgers Discontinuity  |  Test Rel L2: {test_rel_hist[-1]:.4f}')
    plt.tight_layout()
    heatmap_plot = os.path.join(results_dir, 'awfno_burgers_disc_heatmap.png')
    plt.savefig(heatmap_plot, dpi=150)
    plt.close()
    print(f"Spatiotemporal heatmap saved: {heatmap_plot}")

    # --- Plot 3: temporal snapshots (like Figure 5 in the WNO paper) ---
    # Four evenly-spaced snapshots across the predicted horizon.
    snap_indices = [T_OUT // 4 - 1, T_OUT // 2 - 1, 3 * T_OUT // 4 - 1, T_OUT - 1]
    snap_times   = [(idx + T_IN) * 0.02 for idx in snap_indices]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, idx, t_snap in zip(axes, snap_indices, snap_times):
        ax.plot(x_coords, sample_gt[:, idx],   '-',  color='navy',   lw=2, label='Truth')
        ax.plot(x_coords, sample_pred[:, idx], '--', color='crimson', lw=2, label='AW-FNO')
        ax.set_title(f't = {t_snap:.2f} s')
        ax.set_xlabel('x')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('u(x, t)')
    fig.suptitle('AW-FNO Burgers Discontinuity — Temporal Snapshots')
    plt.tight_layout()
    snap_plot = os.path.join(results_dir, 'awfno_burgers_disc_snapshots.png')
    plt.savefig(snap_plot, dpi=150)
    plt.close()
    print(f"Temporal snapshots plot saved: {snap_plot}")


if __name__ == '__main__':
    train_burgers_discontinuity()
