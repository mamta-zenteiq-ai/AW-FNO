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

class SobolevLoss2d(object):
    """
    H1 Sobolev Loss for 2D problems.
    Computes a weighted sum of the relative L2 loss of the values 
    and the relative L2 loss of the gradients (x and y directions).
    """
    def __init__(self, p=2, beta=1.0, eps=1e-8):
        self.p = p
        self.beta = beta  # Weight for the gradient term
        self.eps = eps

    def __call__(self, x, y):
        """
        x: (batch, 1, H, W) or (batch, H, W)
        y: (batch, 1, H, W) or (batch, H, W)
        """
        # Ensure 3D (batch, H, W)
        if x.ndim == 4:
            x = x.squeeze(1)
        if y.ndim == 4:
            y = y.squeeze(1)
        
        # 1. Relative L2 Loss of values
        diff_norm = torch.norm(x - y, self.p, dim=(1, 2))
        y_norm = torch.norm(y, self.p, dim=(1, 2))
        rel_l2 = diff_norm / (y_norm + self.eps)
        
        # 2. Relative L2 Loss of gradients (H1 term)
        # Gradient in x (dim 2)
        dx_x = x[:, :, 1:] - x[:, :, :-1]
        dx_y = y[:, :, 1:] - y[:, :, :-1]
        error_grad_x = torch.norm(dx_x - dx_y, self.p, dim=(1, 2))
        norm_grad_x = torch.norm(dx_y, self.p, dim=(1, 2))
        
        # Gradient in y (dim 1)
        dy_x = x[:, 1:, :] - x[:, :-1, :]
        dy_y = y[:, 1:, :] - y[:, :-1, :]
        error_grad_y = torch.norm(dy_x - dy_y, self.p, dim=(1, 2))
        norm_grad_y = torch.norm(dy_y, self.p, dim=(1, 2))
        
        rel_h1 = (error_grad_x + error_grad_y) / (norm_grad_x + norm_grad_y + self.eps)
        
        # Total Weighted Loss
        return torch.mean(rel_l2 + self.beta * rel_h1)

def train_ns_h1():
    # 1. Configuration
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    epochs = 500
    batch_size = 20
    learning_rate = 1e-3
    print_every = 10
    beta = 0.1  # Weight for H1 derivative term
    
    data_path = '/home/parikshit/AW-FNO/awfno/data/navier_stokes'
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'fno_ns_h1')
    os.makedirs(results_dir, exist_ok=True)
    
    # 2. Load Data
    print("Loading Navier-Stokes (128x128) data for FNO-H1...")
    train_data = torch.load(os.path.join(data_path, 'ns_train_128.pt'))
    test_data = torch.load(os.path.join(data_path, 'ns_test_128.pt'))
    
    x_train = train_data['x'].float()
    y_train = train_data['y'].float()
    x_test = test_data['x'].float()
    y_test = test_data['y'].float()

    if x_train.ndim == 3:
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
        n_modes=(24, 24),
        in_channels=1,
        out_channels=1,
        hidden_channels=64,
        n_layers=4,
        positional_embedding="grid",
        use_channel_mlp=True,
        channel_mlp_dropout=0.0
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    # Use 2D Sobolev (H1) Loss for training
    criterion_h1 = SobolevLoss2d(p=2, beta=beta)
    # Use standard Rel L2 for reporting
    criterion_rel = LpLoss(d=2, p=2, size_average=False)
    
    # 5. Training Loop
    train_loss_history = []
    test_rel_history = []
    
    y_normalizer.to(device)
    
    print(f"Starting FNO 2D training (Sobolev H1 Loss) on Navier-Stokes for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            out = model(batch_x)
            
            # Train using 2D H1 Loss
            loss = criterion_h1(out, batch_y)
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
                
                test_rel += criterion_rel.rel(out, batch_y).item()
                
        test_rel /= len(test_loader.dataset)
        test_rel_history.append(test_rel)
        
        scheduler.step()
        
        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | "
                  f"Train H1 Loss: {train_loss:.8f} | "
                  f"Test Rel L2: {test_rel:.8f}")
            
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")
    print(f"Final Test Relative L2 Error: {test_rel_history[-1]:.8f}")
    
    # 6. Plot Loss History
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train H1 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('H1 Loss')
    plt.title('FNO NS H1 Training')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(test_rel_history, label='Test Rel L2', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2')
    plt.title('FNO NS Test Performance')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fno_ns_h1_loss_plot.png'))
    
    # 7. Visualization of Results
    model.eval()
    with torch.no_grad():
        sample_x, sample_y = next(iter(test_loader))
        sample_x, sample_y = sample_x[0:1].to(device), sample_y[0:1].to(device)
        pred_y = model(sample_x)
        pred_y = y_normalizer.decode(pred_y)
        
        sample_y = sample_y.cpu().numpy().squeeze()
        pred_y = pred_y.cpu().numpy().squeeze()
        
        plt.figure(figsize=(18, 5))
        
        # Ground Truth
        plt.subplot(1, 3, 1)
        plt.imshow(sample_y, cmap='jet')
        plt.colorbar()
        plt.title('Ground Truth')
        
        # Prediction
        plt.subplot(1, 3, 2)
        plt.imshow(pred_y, cmap='jet')
        plt.colorbar()
        plt.title('FNO-H1 Prediction')
        
        # Absolute Error
        plt.subplot(1, 3, 3)
        error = np.abs(sample_y - pred_y)
        plt.imshow(error, cmap='hot')
        plt.colorbar()
        plt.title(f'Absolute Pointwise Error\nMax Error: {np.max(error):.8f}')
        
        plt.tight_layout()
        field_plot_path = os.path.join(results_dir, 'fno_ns_h1_field_comparison.png')
        plt.savefig(field_plot_path)
        print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    train_ns_h1()
