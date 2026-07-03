import torch
import torch.nn as nn
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

from awfno.models.wno import WNO2d
from awfno.utils.unit_gaussian_normalization import UnitGaussianNormalizer
from awfno.utils.losses import LpLoss

def train():
    # 1. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    epochs = 100
    batch_size = 32
    learning_rate = 1e-3
    print_every = 10
    
    data_path = '/media/HDD/mamta_backup/datasets/fno/darcy'
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'wno_only')
    os.makedirs(results_dir, exist_ok=True)
    
    # 2. Load Data
    print("Loading data for WNO training...")
    train_data = torch.load(os.path.join(data_path, 'darcy_train_32.pt'))
    test_data = torch.load(os.path.join(data_path, 'darcy_test_32.pt'))
    
    x_train = train_data['x'].unsqueeze(1).float() # (B, 1, 32, 32)
    y_train = train_data['y'].unsqueeze(1).float() # (B, 1, 32, 32)
    x_test = test_data['x'].unsqueeze(1).float()
    y_test = test_data['y'].unsqueeze(1).float()
    
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
        width=32,
        size=(32, 32),
        level=3,
        n_layers=4,
        padding=0
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion_mse = nn.MSELoss()
    criterion_rel = LpLoss(d=2, p=2, size_average=False)
    
    # 5. Training Loop
    train_mse_history = []
    train_rel_history = []
    test_mse_history = []
    test_rel_history = []
    
    y_normalizer.to(device)
    
    print(f"Starting WNO-only training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_mse = 0.0
        train_rel = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            out = model(batch_x)
            
            loss = criterion_mse(out.view(batch_size, -1), batch_y.view(batch_size, -1))
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
                  f"Train MSE: {train_mse:.6f}, Rel L2: {train_rel:.6f} | "
                  f"Test MSE: {test_mse:.6f}, Rel L2: {test_rel:.6f}")
            
    total_time = time.time() - start_time
    print(f"WNO Training completed in {total_time:.2f}s")
    
    # 6. Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_mse_history, label='Train MSE')
    plt.plot(test_mse_history, label='Test MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('WNO MSE Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_rel_history, label='Train Rel L2')
    plt.plot(test_rel_history, label='Test Rel L2')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2')
    plt.title('WNO Relative L2 Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'wno_loss_plot.png'))
    print(f"Loss plot saved to {os.path.join(results_dir, 'wno_loss_plot.png')}")

if __name__ == "__main__":
    train()
