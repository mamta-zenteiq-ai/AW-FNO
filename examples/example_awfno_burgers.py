import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import sys
import time

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from awfno.models.awfno import AWFNO1d
from awfno.utils.unit_gaussian_normalization import UnitGaussianNormalizer
from awfno.utils.losses import LpLoss
from awfno.utils.seed import set_seed

def train_burgers():
    # 1. Configuration
    set_seed(42)  # — same hyperparameters as FNO baseline for fair comparison
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    epochs = 500
    batch_size = 20
    learning_rate = 1e-3
    print_every = 100

    data_path = '/media/HDD/mamta_backup/datasets/fno/burgers'
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'awfno_burgers')
    os.makedirs(results_dir, exist_ok=True)

    # 2. Load Data
    print("Loading 1D Burgers data...")
    train_data = torch.load(os.path.join(data_path, 'burgers_train_128.pt'))
    test_data  = torch.load(os.path.join(data_path, 'burgers_test_128.pt'))

    x_train = train_data['x'].float()
    y_train = train_data['y'].float()
    x_test  = test_data['x'].float()
    y_test  = test_data['y'].float()

    if x_train.ndim == 2:
        x_train = x_train.unsqueeze(1)
        y_train = y_train.unsqueeze(1)
        x_test  = x_test.unsqueeze(1)
        y_test  = y_test.unsqueeze(1)

    # 3. Normalization (global Gaussian, same as FNO baseline)
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test  = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train_norm = y_normalizer.encode(y_train)

    train_loader = DataLoader(TensorDataset(x_train, y_train_norm), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(x_test, y_test),        batch_size=batch_size, shuffle=False)

    model = AWFNO1d(
        in_channels=1,
        out_channels=1,
        n_modes=(16,),
        size=(128,),
        hidden_channels=64,
        n_layers=4,
        wno_level=3,
        wno_wavelet='db6',
        positional_embedding="grid",   # grid embedding, same as FNO
        non_linearity=F.relu,          # Use ReLU to match FNO
        padding=0,
        dropout=0.0,
        norm=None                      # Disable normalization to match FNO
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Same scheduler as FNO paper: halve LR every 100 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # Train with Relative L2 loss (same as FNO baseline)
    criterion_rel = LpLoss(d=1, p=2, size_average=True)

    # 5. Training Loop
    train_loss_history = []
    test_loss_history  = []

    y_normalizer.to(device)

    print(f"Starting AW-FNO 1D ablation training (Rel L2) on Burgers for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            out = model(batch_x)

            # Train using Relative L2 Loss (normalised space)
            loss = criterion_rel(out.view(out.size(0), -1), batch_y.view(batch_y.size(0), -1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)  # Mean over batches (each already averaged)
        train_loss_history.append(train_loss)

        # Validation
        model.eval()
        test_rel = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = model(batch_x)
                out = y_normalizer.decode(out)

                # Report error on decoded (original scale) data
                test_rel += criterion_rel(out, batch_y).item()

        test_rel /= len(test_loader)  # Mean over batches
        test_loss_history.append(test_rel)

        scheduler.step()

        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | "
                  f"Train Rel L2: {train_loss:.6f} | "
                  f"Test Rel L2: {test_rel:.6f}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")
    print(f"Final Test Relative L2 Error: {test_loss_history[-1]:.6f}")

    # 6. Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Rel L2')
    plt.plot(test_loss_history,  label='Test Rel L2')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Rel L2)')
    plt.title('AW-FNO Burgers Training History')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.semilogy(train_loss_history, label='Train Rel L2')
    plt.semilogy(test_loss_history,  label='Test Rel L2')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('AW-FNO Burgers Training History (Log)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'awfno_burgers_loss_plot.png'))
    print(f"Loss plot saved to {os.path.join(results_dir, 'awfno_burgers_loss_plot.png')}")

    # 7. Visualization of Results (Field Comparison)
    model.eval()
    with torch.no_grad():
        sample_x, sample_y = next(iter(test_loader))
        sample_x, sample_y = sample_x[0:1].to(device), sample_y[0:1].to(device)
        pred_y = model(sample_x)
        pred_y = y_normalizer.decode(pred_y)

        sample_y = sample_y.cpu().numpy().squeeze()
        pred_y   = pred_y.cpu().numpy().squeeze()

        plt.figure(figsize=(8, 5))
        plt.plot(sample_y, label='Ground Truth',      color='blue', linewidth=2)
        plt.plot(pred_y,   '--', label='AW-FNO Prediction', color='red', linewidth=2)
        plt.title(f'AW-FNO Burgers 1D: GT vs Prediction (Rel L2: {test_loss_history[-1]:.4f})')
        plt.xlabel('Spatial Domain')
        plt.ylabel('u(x, T=1)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        field_plot_path = os.path.join(results_dir, 'awfno_burgers_field_comparison.png')
        plt.savefig(field_plot_path)
        print(f"Field comparison plot saved to {field_plot_path}")

if __name__ == "__main__":
    train_burgers()
