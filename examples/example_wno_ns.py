import torch
import torch.nn as nn
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

from awfno.models.wno import WNO2d
from awfno.utils.unit_gaussian_normalization import UnitGaussianNormalizer
from awfno.utils.losses import LpLoss
from awfno.utils.seed import set_seed

def train_ns():
    # 1. Configuration
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    epochs = 500
    batch_size = 20
    learning_rate = 1e-3
    print_every = 10
    
    data_path = '/media/HDD/mamta_backup/datasets/fno/navier_stokes'
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'wno_ns')
    os.makedirs(results_dir, exist_ok=True)
    
    # 2. Load Data
    print("Loading Navier-Stokes (64x64) data for WNO...")
    train_data = torch.load(os.path.join(data_path, 'ns_train_64.pt'))
    test_data = torch.load(os.path.join(data_path, 'ns_test_64.pt'))
    
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
    model = WNO2d(
        in_channels=1,
        out_channels=1,
        width=64, # Matching Burgers experiment capacity
        size=(64, 64),
        level=3,
        n_layers=4,
        padding=0
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    criterion_mse = nn.MSELoss()
    criterion_rel = LpLoss(d=2, p=2, size_average=False)
    
    # 5. Training Loop
    train_mse_history = []
    train_rel_history = []
    test_mse_history = []
    test_rel_history = []
    
    y_normalizer.to(device)
    
    print(f"Starting WNO training on Navier-Stokes for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_mse = 0.0
        train_rel = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            out = model(batch_x)
            
            loss = criterion_mse(out.view(out.size(0), -1), batch_y.view(batch_y.size(0), -1))
            loss.backward()
            optimizer.step()
            
            train_mse += loss.item()
            out_decoded = y_normalizer.decode(out)
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
            print(f"Epoch {epoch}/{epochs} | "
                  f"Train MSE: {train_mse:.8f}, Rel L2: {train_rel:.8f} | "
                  f"Test MSE: {test_mse:.8f}, Rel L2: {test_rel:.8f}")
            
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")
    print(f"Final Test Relative L2 Error: {test_rel_history[-1]:.8f}")
    
    # 6. Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_mse_history, label='Train MSE')
    plt.plot(test_mse_history, label='Test MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('NS WNO MSE Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_rel_history, label='Train Rel L2')
    plt.plot(test_rel_history, label='Test Rel L2')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2')
    plt.title('NS WNO Relative L2 Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'wno_ns_loss_plot.png'))
    print(f"Loss plot saved to {os.path.join(results_dir, 'wno_ns_loss_plot.png')}")
    
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
        plt.title('Prediction')
        
        # Absolute Error
        plt.subplot(1, 3, 3)
        error = np.abs(sample_y - pred_y)
        plt.imshow(error, cmap='hot')
        plt.colorbar()
        plt.title(f'Absolute Pointwise Error\nMax Error: {np.max(error):.8f}')
        
        plt.tight_layout()
        field_plot_path = os.path.join(results_dir, 'wno_ns_field_comparison.png')
        plt.savefig(field_plot_path)
        print(f"Field plot saved to {field_plot_path}")

if __name__ == "__main__":
    train_ns()
