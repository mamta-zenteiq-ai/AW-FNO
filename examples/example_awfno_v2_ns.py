"""
Training script for AW-FNO v2 on the 2-D Navier-Stokes equation.

Uses the same data pipeline, optimizer, scheduler, loss and evaluation
as the FNO / WNO / AW-FNO v1 NS scripts for a fair comparison.
Training recipe follows the FNO paper: 500 epochs, StepLR (halve every 100).
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

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from awfno.models.awfno_v2 import AWFNOv2_3d
from awfno.utils.unit_gaussian_normalization import UnitGaussianNormalizer
from awfno.utils.losses import LpLoss


def train_ns():
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

    epochs = 100
    batch_size = 20
    learning_rate = 1e-3
    print_every = 50

    data_path = '/media/HDD/mamta_backup/datasets/fno/navier_stokes'
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'awfno_v2_ns')
    os.makedirs(results_dir, exist_ok=True)

    # ─── 2. Load Data ────────────────────────────────────────────────
    print("Loading Navier-Stokes (64x64) data...")
    train_data = torch.load(os.path.join(data_path, 'ns_train_64.pt'))
    test_data  = torch.load(os.path.join(data_path, 'ns_test_64.pt'))

    x_train = train_data['x'].float()
    y_train = train_data['y'].float()
    x_test  = test_data['x'].float()
    y_test  = test_data['y'].float()

    if x_train.ndim == 3:
        x_train = x_train.unsqueeze(1).unsqueeze(2) # [B, C, T, H, W] = [1000, 1, 1, 64, 64]
        y_train = y_train.unsqueeze(1).unsqueeze(2)
        x_test  = x_test.unsqueeze(1).unsqueeze(2)
        y_test  = y_test.unsqueeze(1).unsqueeze(2)

    # ─── 3. Normalization ────────────────────────────────────────────
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test  = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train_norm = y_normalizer.encode(y_train)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train_norm),
        batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size, shuffle=False,
    )

    # ─── 4. Model, Optimizer, Loss ───────────────────────────────────
    model = AWFNOv2_3d(
        in_channels=1,
        out_channels=1,
        n_modes=(1, 12, 12), # Added T modes = 1
        size=(1, 64, 64),    # Added T size = 1
        hidden_channels=16, 
        n_fno_layers=4,
        n_wno_layers=4,
        padding=0,
        dropout=0.0,
        wno_wavelet='db6',
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"AWFNOv2_3d  —  trainable parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    criterion_mse = nn.MSELoss()
    criterion_rel = LpLoss(d=3, p=2, size_average=False) # Changed d to 3

    # ─── 5. Training Loop ────────────────────────────────────────────
    train_mse_history = []
    train_rel_history = []
    test_mse_history  = []
    test_rel_history  = []

    y_normalizer.to(device)

    print(f"Starting AW-FNO v2 training on Navier-Stokes for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_mse = 0.0
        train_rel = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            out = model(batch_x)

            loss = criterion_mse(
                out.view(out.size(0), -1),
                batch_y.view(batch_y.size(0), -1),
            )
            loss.backward()
            optimizer.step()

            train_mse += loss.item()
            out_decoded     = y_normalizer.decode(out)
            batch_y_decoded = y_normalizer.decode(batch_y)
            train_rel += criterion_rel.rel(out_decoded, batch_y_decoded).item()

        train_mse /= len(train_loader)
        train_rel /= len(train_loader.dataset)

        train_mse_history.append(train_mse)
        train_rel_history.append(train_rel)

        # Validation
        model.eval()
        test_mse = 0.0
        test_rel = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = model(batch_x)
                out = y_normalizer.decode(out)

                test_mse += criterion_mse(out, batch_y).item()
                test_rel += criterion_rel.rel(out, batch_y).item()

        test_mse /= len(test_loader)
        test_rel /= len(test_loader.dataset)

        test_mse_history.append(test_mse)
        test_rel_history.append(test_rel)

        scheduler.step()

        if epoch % print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train MSE: {train_mse:.6f}, Rel L2: {train_rel:.6f} | "
                f"Test  MSE: {test_mse:.6f}, Rel L2: {test_rel:.6f}"
            )

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")

    # ─── 6. Plot Loss ────────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_mse_history, label='Train MSE')
    plt.plot(test_mse_history,  label='Test MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('AW-FNO v2 NS MSE Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_rel_history, label='Train Rel L2')
    plt.plot(test_rel_history,  label='Test Rel L2')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2')
    plt.title('AW-FNO v2 NS Relative L2 Loss')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'awfno_v2_ns_loss_plot.png')
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")

    # Save model
    model_path = os.path.join(results_dir, 'awfno_v2_ns_best.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_ns()
