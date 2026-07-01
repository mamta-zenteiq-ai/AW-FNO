"""
2D Darcy flow in triangular domain with notch — FNO version with H1 Sobolev loss.

Paper reference: WNO paper (Tripura & Chakraborty, CMA 2023), Section 4.3.

PDE:
    -∇·(a(x,y)∇u(x,y)) = f(x,y),  x,y ∈ ω  (triangular domain with notch)
    u(x,y) = u_bc(x,y),             x,y ∈ ∂ω  (boundary condition from GP)
    a(x,y) = 0.1,  f(x,y) = -1

Operator being learned:
    D: u(x,y)|_{∂ω} → u(x,y)
    i.e., boundary conditions → full pressure field over the triangular domain.

Dataset: Darcy_Triangular_FNO.mat
  - boundCoeff : (2000, 101, 101) — boundary coefficient field (input)
  - sol        : (2000, 101, 101) — full solution field (output)
  Subsampled to 51×51 (stride-2 indexing [::2, ::2]).
  Train: 1900 samples, Test: 100 samples.

Model: FNO (Fourier Neural Operator)
  n_modes=(12,12), hidden_channels=64, n_layers=4, epochs=500.

Loss: H1 Sobolev loss (relative-L2 value loss + beta * relative-L2 gradient loss).

Per-epoch log: time | Train MSE | Train H1 | Train Rel-L2 | Test MSE | Test H1 | Test Rel-L2
Results saved to: results/fno_darcy_notch/
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
    print("WARNING: scipy not found — cannot load .mat files.")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from awfno.models.fno import FNO
from awfno.utils.unit_gaussian_normalization import UnitGaussianNormalizer
from awfno.utils.losses import LpLoss
from awfno.utils.seed import set_seed


# ===========================================================================
# H1 Sobolev loss for 2D fields  (B, 1, H, W) or (B, H, W)
# ===========================================================================
class SobolevLoss2d:
    """
    H1 Sobolev loss for 2D spatial fields.

    loss = mean_over_batch(
        rel_L2(u, u_gt)
        + beta * [ rel_L2(∂_x u, ∂_x u_gt) + rel_L2(∂_y u, ∂_y u_gt) ]
    )

    Finite differences along axis -1 (x) and axis -2 (y).
    """

    def __init__(self, p: int = 2, beta: float = 0.1, eps: float = 1e-8):
        self.p    = p
        self.beta = beta
        self.eps  = eps

    def _rel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Relative Lp norm averaged over batch."""
        a_flat = a.reshape(a.size(0), -1)
        b_flat = b.reshape(b.size(0), -1)
        diff   = torch.norm(a_flat - b_flat, self.p, dim=1)
        norm_b = torch.norm(b_flat,          self.p, dim=1)
        return torch.mean(diff / (norm_b + self.eps))

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Squeeze channel dim if present
        if pred.ndim == 4:
            pred   = pred.squeeze(1)
        if target.ndim == 4:
            target = target.squeeze(1)

        # Value loss
        loss = self._rel(pred, target)

        # Gradient along x (last axis)
        dx_pred   = pred[..., 1:]   - pred[..., :-1]
        dx_target = target[..., 1:] - target[..., :-1]
        loss = loss + self.beta * self._rel(dx_pred, dx_target)

        # Gradient along y (second-to-last axis)
        dy_pred   = pred[..., 1:, :]   - pred[..., :-1, :]
        dy_target = target[..., 1:, :] - target[..., :-1, :]
        loss = loss + self.beta * self._rel(dy_pred, dy_target)

        return loss


# ===========================================================================
# Helpers
# ===========================================================================

def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"


def load_darcy_triangular(mat_path: str, subsample: int = 2):
    """
    Load Darcy triangular data from Darcy_Triangular_FNO.mat.

    Returns
    -------
    x_data : (N, 1, H, W)  — boundary coefficient field (input)
    y_data : (N, 1, H, W)  — solution field (output)
    """
    assert SCIPY_AVAILABLE, "scipy is required to load .mat files."
    raw = loadmat(mat_path)

    bc  = raw['boundCoeff']   # (N, 101, 101)
    sol = raw['sol']          # (N, 101, 101)

    # Subsample from 101x101 → 51x51
    bc  = bc[:,  ::subsample, ::subsample]
    sol = sol[:, ::subsample, ::subsample]

    x_data = torch.tensor(bc,  dtype=torch.float32).unsqueeze(1)   # (N, 1, 51, 51)
    y_data = torch.tensor(sol, dtype=torch.float32).unsqueeze(1)   # (N, 1, 51, 51)
    return x_data, y_data


# ===========================================================================
# Main training function
# ===========================================================================

def train_fno_darcy_notch():
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # Hyperparameters
    # -----------------------------------------------------------------------
    epochs      = 500
    batch_size  = 20
    lr          = 1e-3
    beta_h1     = 0.1       # derivative weight in H1 loss
    n_train     = 1900
    n_test      = 100
    subsample   = 2         # 101 → 51 per spatial dimension

    # -----------------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------------
    data_dir    = os.path.join(PROJECT_ROOT, 'awfno', 'data', 'darcy')
    mat_file    = os.path.join(data_dir, 'Darcy_Triangular_FNO.mat')
    pt_file     = os.path.join(data_dir, 'darcy_triangular_51.pt')
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'fno_darcy_notch')
    os.makedirs(results_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load / cache data
    # -----------------------------------------------------------------------
    print("Loading Darcy triangular (notch) data...")
    if os.path.exists(pt_file):
        saved = torch.load(pt_file)
        x_data, y_data = saved['x'], saved['y']
        print(f"  Loaded from cache: {pt_file}  (shape {tuple(x_data.shape)})")
    elif os.path.exists(mat_file):
        x_data, y_data = load_darcy_triangular(mat_file, subsample=subsample)
        torch.save({'x': x_data, 'y': y_data}, pt_file)
        print(f"  Loaded from .mat, cached to {pt_file}  (shape {tuple(x_data.shape)})")
    else:
        raise FileNotFoundError(
            f"Data file not found.\n"
            f"  Expected .pt : {pt_file}\n"
            f"  Expected .mat: {mat_file}"
        )

    # x_data: (2000, 1, 51, 51),  y_data: (2000, 1, 51, 51)
    N, _, H, W = x_data.shape
    assert N >= n_train + n_test, f"Not enough samples: {N} < {n_train + n_test}"
    print(f"  Total {N} samples  |  Grid {H}×{W}  |  Split {n_train}/{n_test}")

    x_train = x_data[:n_train]
    y_train = y_data[:n_train]
    x_test  = x_data[n_train : n_train + n_test]
    y_test  = y_data[n_train : n_train + n_test]

    # -----------------------------------------------------------------------
    # Normalisation
    # -----------------------------------------------------------------------
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train_n    = x_normalizer.encode(x_train)
    x_test_n     = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train_n    = y_normalizer.encode(y_train)
    # y_test kept in physical units for evaluation

    train_loader = DataLoader(
        TensorDataset(x_train_n, y_train_n),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test_n, y_test),
        batch_size=batch_size, shuffle=False
    )

    # -----------------------------------------------------------------------
    # Model — FNO
    # n_modes=(12,12): keeps up to 12 Fourier modes per spatial dim (≤ 51//2=25)
    # hidden_channels=64, n_layers=4 consistent with WNO paper Table 1 settings
    # -----------------------------------------------------------------------
    model = FNO(
        n_modes=(12, 12),
        in_channels=1,
        out_channels=1,
        hidden_channels=192,
        n_layers=4,
        positional_embedding="grid",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    criterion_h1  = SobolevLoss2d(beta=beta_h1)
    criterion_mse = nn.MSELoss()
    criterion_rel = LpLoss(d=2, p=2, size_average=True)

    x_normalizer.to(device)
    y_normalizer.to(device)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    train_h1_hist  = []
    train_mse_hist = []
    train_rel_hist = []
    test_h1_hist   = []
    test_mse_hist  = []
    test_rel_hist  = []

    header = (
        f"{'Epoch':>6} | {'H1 Tr':>10} | {'MSE Tr':>10} | {'RelL2 Tr':>9} | "
        f"{'H1 Te':>10} | {'MSE Te':>10} | {'RelL2 Te':>9} | {'t_ep':>7} | {'ETA':>10}"
    )
    print(f"\nStarting FNO Darcy-notch training for {epochs} epochs...")
    print(header)
    print("-" * len(header))

    log_path  = os.path.join(results_dir, 'training_log.txt')
    log_lines = [header + "\n"]

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        t_ep = time.time()

        # ---- Training ----
        model.train()
        tr_h1 = tr_mse = tr_rel = 0.0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()

            pred = model(bx)                          # (B, 1, 51, 51)

            # Decode to physical space before all loss computations
            pred_dec = y_normalizer.decode(pred)
            by_dec   = y_normalizer.decode(by)

            # H1 Sobolev loss drives backprop
            loss = criterion_h1(pred_dec, by_dec)
            loss.backward()
            optimizer.step()

            tr_h1  += loss.item()
            with torch.no_grad():
                tr_mse += criterion_mse(pred_dec, by_dec).item()
                tr_rel += criterion_rel(
                    pred_dec.view(pred_dec.size(0), -1),
                    by_dec.view(by_dec.size(0), -1)
                ).item()

        n_tr = len(train_loader)
        tr_h1  /= n_tr;  tr_mse /= n_tr;  tr_rel /= n_tr
        train_h1_hist.append(tr_h1)
        train_mse_hist.append(tr_mse)
        train_rel_hist.append(tr_rel)

        # ---- Evaluation ----
        model.eval()
        te_h1 = te_mse = te_rel = 0.0

        with torch.no_grad():
            for bx_n, by_phys in test_loader:
                bx_n, by_phys = bx_n.to(device), by_phys.to(device)

                pred_n   = model(bx_n)
                pred_dec = y_normalizer.decode(pred_n)

                te_h1  += criterion_h1(pred_dec, by_phys).item()
                te_mse += criterion_mse(pred_dec, by_phys).item()
                te_rel += criterion_rel(
                    pred_dec.view(pred_dec.size(0), -1),
                    by_phys.view(by_phys.size(0), -1)
                ).item()

        n_te = len(test_loader)
        te_h1 /= n_te;  te_mse /= n_te;  te_rel /= n_te
        test_h1_hist.append(te_h1)
        test_mse_hist.append(te_mse)
        test_rel_hist.append(te_rel)

        scheduler.step()

        # ---- Logging ----
        ep_time = time.time() - t_ep
        elapsed = time.time() - t0
        eta     = elapsed / epoch * (epochs - epoch)

        line = (
            f"{epoch:>6}/{epochs} | {tr_h1:>10.4e} | {tr_mse:>10.4e} | {tr_rel:>9.4f} | "
            f"{te_h1:>10.4e} | {te_mse:>10.4e} | {te_rel:>9.4f} | "
            f"{ep_time:>6.1f}s | {_fmt_time(eta):>10}"
        )
        print(line)
        log_lines.append(line + "\n")

    elapsed_total = time.time() - t0
    summary = (
        f"\nTraining completed in {_fmt_time(elapsed_total)}  |  "
        f"Final Test Rel-L2: {test_rel_hist[-1]:.6f}  |  "
        f"Final Test H1: {test_h1_hist[-1]:.6e}\n"
    )
    print(summary)
    log_lines.append(summary)

    with open(log_path, 'w') as f:
        f.writelines(log_lines)
    print(f"Training log saved: {log_path}")

    # -----------------------------------------------------------------------
    # Save model checkpoint
    # -----------------------------------------------------------------------
    ckpt_path = os.path.join(results_dir, 'fno_darcy_notch_model.pt')
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model checkpoint saved: {ckpt_path}")

    # -----------------------------------------------------------------------
    # Plot 1 — Loss curves (3 panels: H1, MSE, Rel-L2)
    # -----------------------------------------------------------------------
    epochs_ax = range(1, epochs + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].semilogy(epochs_ax, train_h1_hist,  label='Train H1 (Sobolev)', color='purple')
    axes[0].semilogy(epochs_ax, test_h1_hist,   label='Test H1 (Sobolev)',  color='orchid', linestyle='--')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('H1 Loss')
    axes[0].set_title('FNO Darcy Notch — Sobolev H1 Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(epochs_ax, train_mse_hist, label='Train MSE', color='steelblue')
    axes[1].semilogy(epochs_ax, test_mse_hist,  label='Test MSE',  color='steelblue', linestyle='--')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MSE')
    axes[1].set_title('FNO Darcy Notch — MSE')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(epochs_ax, train_rel_hist, label='Train Rel-L2', color='tomato')
    axes[2].semilogy(epochs_ax, test_rel_hist,  label='Test Rel-L2',  color='tomato', linestyle='--')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Relative L2')
    axes[2].set_title('FNO Darcy Notch — Relative L2')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        f'FNO Darcy Triangular (Notch)  |  '
        f'Final Test Rel-L2: {test_rel_hist[-1]:.4f}  |  '
        f'Final Test H1: {test_h1_hist[-1]:.4e}',
        fontsize=11
    )
    plt.tight_layout()
    loss_plot = os.path.join(results_dir, 'fno_darcy_notch_loss.png')
    plt.savefig(loss_plot, dpi=150)
    plt.close()
    print(f"Loss plot saved: {loss_plot}")

    # -----------------------------------------------------------------------
    # Plot 2 — GT | Prediction | Error for 4 test samples  (like Fig 7 WNO)
    # -----------------------------------------------------------------------

    def apply_domain_mask(ax, s=1):
        """
        Whiteout the three corners outside the triangular domain and the
        vertical slit (notch) centred at x=0.5 that runs from y=0 to y≈0.41.

        The triangular domain has vertices at (0,0), (1,0), and (0.5, ~0.84).
        The grid stores values for the full unit square; points outside the
        triangle are set to zero by the solver, so we paint them white here
        to expose the correct triangular shape.

        The notch is a thin vertical slit at x≈0.5 from y=0 to y≈0.41
        (≈ 21/51 of the grid height).  We paint it white via a Rectangle patch.
        """
        from matplotlib.patches import Rectangle

        ymax = s - 8 / 51           # triangle apex y-coordinate (~0.843)

        # Left corner (below the left edge of the triangle)
        xf = np.array([0., s / 2])
        yf = xf * (ymax / (s / 2))
        ax.fill_between(xf, yf, ymax, color='white')

        # Right corner (below the right edge of the triangle)
        xf = np.array([s / 2, s])
        yf = (xf - s) * (ymax / (s / 2 - s))
        ax.fill_between(xf, yf, ymax, color='white')

        # Top strip (above the apex)
        ax.fill_between([0, s], ymax, s, color='white')

        # Notch: thin vertical slit centred at x=0.5, height ≈ 0.41
        ax.add_patch(Rectangle((0.49, 0), 0.02, 0.41,
                                facecolor='white', zorder=5))

    # Spread sample indices across the full test set (matching reference script)
    vis_indices = [1, 30, 59, 88]
    n_vis = len(vis_indices)

    # Collect all test predictions in one forward pass (batched)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for bx_n, _ in test_loader:
            bx_n = bx_n.to(device)
            out  = y_normalizer.decode(model(bx_n)).cpu()
            all_preds.append(out)
    all_preds = torch.cat(all_preds, dim=0)   # (n_test, 1, H, W)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size']   = 13

    fig, axes = plt.subplots(n_vis, 3, figsize=(15, 4 * n_vis))
    col_titles = ['Ground Truth', 'FNO Prediction', 'Absolute Error']

    for row, idx in enumerate(vis_indices):
        gt  = y_test[idx].squeeze().numpy()            # (H, W)
        pr  = all_preds[idx].squeeze().numpy()         # (H, W)
        err = np.abs(gt - pr)

        rel_err = np.linalg.norm(gt - pr) / (np.linalg.norm(gt) + 1e-8)

        for col_i, (field, cmap, ctitle) in enumerate(zip(
            [gt, pr, err],
            ['nipy_spectral', 'nipy_spectral', 'jet'],
            col_titles,
        )):
            ax = axes[row, col_i]
            im = ax.imshow(
                field,
                origin='lower',
                extent=[0, 1, 0, 1],          # proper x,y ∈ [0,1] axis labels
                interpolation='Gaussian',      # smooth rendering at 51×51
                aspect='equal',               # triangle looks correct (not squashed)
                cmap=cmap,
            )
            plt.colorbar(im, ax=ax, fraction=0.045)
            if row == 0:
                ax.set_title(ctitle, fontsize=14, fontweight='bold')
            ax.set_xlabel('x', fontweight='bold')
            ax.set_ylabel('y', fontweight='bold')
            apply_domain_mask(ax)             # triangular domain + notch mask

        # Row label: sample index + rel L2
        axes[row, 0].set_ylabel(
            f'Sample {idx + 1}\n(RelL2={rel_err:.4f})\ny',
            fontweight='bold'
        )

    fig.suptitle(
        f'FNO — Darcy Triangular Notch: GT | Prediction | Error\n'
        f'Mean Test Rel-L2: {test_rel_hist[-1]:.4f}',
        fontsize=14
    )
    plt.tight_layout()
    field_plot = os.path.join(results_dir, 'fno_darcy_notch_fields.png')
    plt.savefig(field_plot, dpi=150, bbox_inches='tight')
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)   # restore defaults
    print(f"Field comparison plot saved: {field_plot}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  FNO Darcy Notch — Training Summary")
    print(f"{'='*70}")
    print(f"  Epochs           : {epochs}")
    print(f"  Train samples    : {n_train}")
    print(f"  Test samples     : {n_test}")
    print(f"  Grid size        : {H}×{W}")
    print(f"  Model parameters : {n_params:,}")
    print(f"  Final Train H1   : {train_h1_hist[-1]:.6e}")
    print(f"  Final Train MSE  : {train_mse_hist[-1]:.6e}")
    print(f"  Final Train Rel-L2: {train_rel_hist[-1]:.6f}")
    print(f"  Final Test H1    : {test_h1_hist[-1]:.6e}")
    print(f"  Final Test MSE   : {test_mse_hist[-1]:.6e}")
    print(f"  Final Test Rel-L2: {test_rel_hist[-1]:.6f}")
    print(f"  Training time    : {_fmt_time(elapsed_total)}")
    print(f"  Results saved to : {results_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    train_fno_darcy_notch()
