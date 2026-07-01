import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from awfno.models.fno import FNO
from awfno.utils.unit_gaussian_normalization import UnitGaussianNormalizer
from awfno.utils.losses import LpLoss
from awfno.utils.seed import set_seed

class SobolevLoss(object):
    """
    H1 Sobolev Loss for 1D problems.
    Computes a weighted sum of the relative L2 loss of the values 
    and the relative L2 loss of the first-order derivatives.
    """
    def __init__(self, p=2, beta=1.0, eps=1e-8):
        self.p = p
        self.beta = beta  # Weight for the derivative term
        self.eps = eps

    def __call__(self, x, y):
        """
        x: (batch, spatial_dim)
        y: (batch, spatial_dim)
        """
        # Ensure 2D (batch, N)
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        
        # 1. Relative L2 Loss of values
        diff_norm = torch.norm(x - y, self.p, dim=1)
        y_norm = torch.norm(y, self.p, dim=1)
        rel_l2 = diff_norm / (y_norm + self.eps)
        
        # 2. Relative L2 Loss of derivatives (H1 term)
        # Using finite differences
        dx_x = x[:, 1:] - x[:, :-1]
        dx_y = y[:, 1:] - y[:, :-1]
        
        diff_grad_norm = torch.norm(dx_x - dx_y, self.p, dim=1)
        y_grad_norm = torch.norm(dx_y, self.p, dim=1)
        rel_h1 = diff_grad_norm / (y_grad_norm + self.eps)
        
        # Total Weighted Loss
        return torch.mean(rel_l2 + self.beta * rel_h1)

def train_burgers_h1():
    # 1. Configuration
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    epochs = 500
    batch_size = 20
    learning_rate = 1e-3
    print_every = 100
    beta = 0.1  # Weight for H1 derivative term
    
    data_path = '/media/HDD/mamta_backup/datasets/fno/burgers'
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'fno_burgers_h1')
    os.makedirs(results_dir, exist_ok=True)
    
    # 2. Load Data
    print("Loading 1D Burgers data...")
    train_data = torch.load(os.path.join(data_path, 'burgers_train_128.pt'))
    test_data = torch.load(os.path.join(data_path, 'burgers_test_128.pt'))
    
    x_train = train_data['x'].float()
    y_train = train_data['y'].float()
    x_test = test_data['x'].float()
    y_test = test_data['y'].float()

    if x_train.ndim == 2:
        x_train = x_train.unsqueeze(1)
        y_train = y_train.unsqueeze(1)
        x_test = x_test.unsqueeze(1)
        y_test = y_test.unsqueeze(1)
    
    # 3. Normalization
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train_norm = y_normalizer.encode(y_train)
    
    train_loader = DataLoader(TensorDataset(x_train, y_train_norm), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    # 4. Model, Optimizer, Loss
    model = FNO(
        n_modes=(16,),
        in_channels=1,
        out_channels=1,
        hidden_channels=64,
        n_layers=4,
        positional_embedding="grid",
        non_linearity=F.relu,
        norm=None,
        use_channel_mlp=False
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    # Use Sobolev (H1) Loss for training
    criterion_h1 = SobolevLoss(p=2, beta=beta)
    # Use standard Rel L2 for reporting/comparison
    criterion_rel = LpLoss(d=1, p=2, size_average=True)
    
    # 5. Training Loop
    train_loss_history = []
    test_rel_history = []
    
    y_normalizer.to(device)
    
    print(f"Starting FNO 1D training (Sobolev H1 Loss) on Burgers for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            out = model(batch_x)
            
            # Train using H1 Sobolev Loss
            # out shape: (B, 1, L) -> flatten to (B, L)
            loss = criterion_h1(out.view(out.size(0), -1), batch_y.view(batch_y.size(0), -1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_loss_history.append(train_loss)
        
        # Validation
        model.eval()
        test_rel = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = model(batch_x)
                out = y_normalizer.decode(out)
                
                # Report standard Rel L2 error on decoded data
                test_rel += criterion_rel(out, batch_y).item()
                
        test_rel /= len(test_loader)
        test_rel_history.append(test_rel)
        
        scheduler.step()
        
        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | "
                  f"Train H1 Loss: {train_loss:.6f} | "
                  f"Test Rel L2: {test_rel:.6f}")
            
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")
    print(f"Final Test Relative L2 Error: {test_rel_history[-1]:.6f}")
    
    # 6. Plot Results
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(train_loss_history, label='Train H1 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('H1 Loss')
    plt.title('FNO Burgers H1 Training Loss')
    plt.legend()
    
    # Test error plot
    plt.subplot(1, 3, 2)
    plt.plot(test_rel_history, label='Test Rel L2', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2')
    plt.title('FNO Burgers Test Performance')
    plt.legend()
    
    # Log scale loss
    plt.subplot(1, 3, 3)
    plt.semilogy(train_loss_history, label='Train H1 Loss')
    plt.semilogy(test_rel_history, label='Test Rel L2')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log)')
    plt.title('Training History (Log Scale)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fno_burgers_h1_loss_plot.png'))
    print(f"Results plot saved to {os.path.join(results_dir, 'fno_burgers_h1_loss_plot.png')}")
    
    # 7. Visualization of Results (Field Comparison)
    model.eval()
    with torch.no_grad():
        sample_x, sample_y = next(iter(test_loader))
        sample_x, sample_y = sample_x[0:1].to(device), sample_y[0:1].to(device)
        pred_y = model(sample_x)
        pred_y = y_normalizer.decode(pred_y)
        
        sample_y = sample_y.cpu().numpy().squeeze()
        pred_y = pred_y.cpu().numpy().squeeze()
        
        plt.figure(figsize=(14, 5))
        
        # Subplot 1: Field Comparison
        plt.subplot(1, 2, 1)
        plt.plot(sample_y, label='Ground Truth', color='blue', linewidth=2)
        plt.plot(pred_y, '--', label='FNO-H1 Prediction', color='red', linewidth=2)
        plt.title(f'FNO Burgers 1D (H1 Loss): GT vs Prediction\nTest Rel L2: {test_rel_history[-1]:.6f}')
        plt.xlabel('Spatial Domain')
        plt.ylabel('u(x, T=1)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Pointwise Error
        plt.subplot(1, 2, 2)
        error = np.abs(sample_y - pred_y)
        plt.plot(error, color='green', linewidth=2, label='Abs Error')
        plt.fill_between(range(len(error)), error, alpha=0.2, color='green')
        plt.title(f'Pointwise Absolute Error\nMax Error: {np.max(error):.6f}')
        plt.xlabel('Spatial Domain')
        plt.ylabel('|Error|')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        field_plot_path = os.path.join(results_dir, 'fno_burgers_h1_field_comparison.png')
        plt.savefig(field_plot_path)
        print(f"Field comparison with error plot saved to {field_plot_path}")

if __name__ == "__main__":
    train_burgers_h1()
