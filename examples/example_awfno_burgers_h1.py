import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os 
'''
os is used to manage things outside of python - the operating system. 
os(operating system) module is used to interact with the file system, allowing us to create directories, save files, 
and manage paths in a platform-independent way (works for different operating systems like Linux, Windows, macOS).

 In this code, we use os to create a results directory and save plots.
'''
import sys
'''
sys(system) means the python runtime environment itself - the software that executes the python code - which acts as 
the bridge between the code we write (python script) and the underlying operating system.
The sys module provides access to some variables used or maintained by the Python interpreter and to functions that 
interact strongly with the python interpreter. It does not care about the your hardware specs or how files are arranged 
on your hard drive, it looks inward at the python process itself. It controls 1) what folders are allowed to be imported 
as modules in python (using sys.path), and it allows us to modify that list of folders at runtime. 2) How should python 
exit if program crashes or finishes (using sys.exit). 3) What version of python is running the code (using sys.version). 
4) What command line arguments were passed while launching the python script (using sys.argv). 
5) How to handle uncaught exceptions (using sys.excepthook).

Here, we use sys.path to add the project root directory to the Python path,
allowing us to import modules from the awfno package.
'''
import time
'''
This is used to track how long the training process takes. We record the start time before training and then calculate 
the total time taken after training completes. This helps us understand the computational cost of training the model. 
'''
import numpy as np  # scientific computing library for handling arrays and numerical operations.

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
'''
1) __file__ : This is a special/ hidden variable in Python that contains the path to the current script. 
# Here: __file__  = /home/gazania/zan_folder/AW-FNO/examples/example_awfno_burgers_h1.py --> absolute path 
2) os.path.abspath(__file__) makes sure we have the absolute path (from root partition (/)) to the current script.
3) os.path.dirname(...) gives us the directory containing the current script. It removes the last part of the path
(the script name) and gives us the directory. So we get /home/gazania/zan_folder/AW-FNO/examples
4) os.path.dirname(...) again gives us the parent directory, removing the 'examples' part,
so we get /home/gazania/zan_folder/AW-FNO, which is the "PROJECT_ROOT".
'''
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) # This adds the "PROJECT_ROOT" to the list of directories that Python searches 
    # when we do an import. By inserting it at index 0, we ensure that it is at the first position. 
    '''
    When python reaches from awfno.models.awfno import AWFNO1d, it looks through the directories in sys.path in order.
    If it finds awfno package in the "PROJECT_ROOT" directory, it will import AWFNO1d from awfno.models.awfno.
    '''


from awfno.models.awfno import AWFNO1d
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

def train_burgers_awfno_h1():
    # 1. Configuration
    set_seed(42)  # same hyperparameters as FNO baseline for fair comparison.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    epochs = 500
    batch_size = 20
    learning_rate = 1e-3
    print_every = 100
    beta = 0.1  # Weight for H1 derivative term
    
    data_path = '/home/parikshit/AW-FNO/awfno/data/burgers'  
    '''
    This is hardcoded for the particular euclid system. 
    It points to the location where the preprocessed Burgers dataset is stored. 
    You may need to change this path to where you have the dataset on your system.
    '''

    results_dir = os.path.join(PROJECT_ROOT, 'results', 'awfno_burgers_h1')
    '''
    This constructs a path to save our output results. (like loss plots and field comparison plots).
    It creates dynamically a "results" folder in the "PROJECT_ROOT" and then a subfolder "awfno_burgers_h1" for this specific 
    experiment. So, results_dir = /home/gazania/zan_folder/AW-FNO/results/awfno_burgers_h1
    '''

    os.makedirs(results_dir, exist_ok=True)
    '''
    This line physically creates the directory specified by results_dir in the file system if it does not already exist.
    The "exist_ok=True" argument means that if the directory already exists (bcz of the previous runs),
    it will not raise an error and will simply do nothing.
    '''
    
    # 2. Load Data
    print("Loading 1D Burgers data...")
    train_data = torch.load(os.path.join(data_path, 'burgers_train_128.pt'))
    test_data = torch.load(os.path.join(data_path, 'burgers_test_128.pt'))
    '''
    1) .pt files are PyTorch's way of saving tensors and other objects like python dictionaries. 
    These .pt (.pth) files are binary files (collection of 0s and 1s in a such way that they can be loaded 
    back as tensors into the memory).
    2) Here, we load the training and testing datasets for the 1D Burgers problem. Each of these files contains a 
    dictionary with keys 'x' and 'y', where 'x' is the input (initial fluid velocity field at t=0) and 
    'y' is the output (fluid velocity field at final time t=1 or t=T).
    3) torch.load() reads and deserializes(deserialization means converting from binary format back 
    to Python objects (like dictionaries, lists, etc.)) the .pt file and gives us the original data structure
    (in this case, a dictionary with tensors).
    '''
    
    x_train = train_data['x'].float() # Extract the input tensors from the training data by using the key 'x' 
                                      # and convert it to (torch.float32) float type (32-bit floating point). 
                                      # This is standard precision for training neural networks, as it provides a good
                                      # balance between numerical precision and memory efficiency.
    y_train = train_data['y'].float()
    x_test = test_data['x'].float()
    y_test = test_data['y'].float()
    print(f"Loaded data shapes - x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}")

    if x_train.ndim == 2:  
        '''
        .ndim is attribute in PyTorch that gives the number of dimensions of the tensor.
        Here, we check if the input tensors are 2D (batch_size, spatial_grid(actually values at each grid point))).
        For 128 grid points, the shape would be (num_samples, 128).
        If the data is 2D, we need to add a channel dimension to make it compatible with the model which expects
        (batch_size, in_channels, spatial_grid). For 1D problems, in_channels = 1 (typically).
        '''
        x_train = x_train.unsqueeze(1) # This adds a new dimension at index 1 (the channel dimension), 
                                       # so the shape changes from (num_samples, 128) to (num_samples, 1, 128).
        y_train = y_train.unsqueeze(1)
        x_test = x_test.unsqueeze(1)
        y_test = y_test.unsqueeze(1)
    
    # 3. Normalization
    x_normalizer = UnitGaussianNormalizer(x_train)
    '''
    This creates an instance of the UnitGaussianNormalizer class for the input data (x_train).
    The normalizer computes the mean and standard deviation of the training data.
    '''
    x_train = x_normalizer.encode(x_train)
    '''
    This applies the normalization to the training data. The "encode()" method transforms the data to have 
    zero mean and unit variance based on the statistics computed from the training data. 
    Normalization is crucial for training neural networks as it helps with convergence and stability.
    '''
    x_test = x_normalizer.encode(x_test)
    '''
    We also apply the same normalization to the test data using the same normalizer instance which has the mean and
    std computed from the training data. In NN, we must never use mean and std from test data for normalization, as
    it would lead to data leakage to the model.
    '''
    
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train_norm = y_normalizer.encode(y_train)
    '''
    The original output data or ground truth (y_test) is not normalized, as we want to report the final test error 
    in the original physical units (unscaled and in accordance with the real-world scale) for better interpretability 
    and comparison with other methods.
    '''
    
    train_loader = DataLoader(TensorDataset(x_train, y_train_norm), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False) 
    '''
    For evaluation, we set shuffle=False to maintain the order of the test samples, which can be useful 
    for certain types of analysis or visualization. For training, we set shuffle=True to ensure that the model
    sees the data in a different order each epoch, which can help with generalization and prevent overfitting.
    '''

    model = AWFNO1d(
        in_channels=1,
        out_channels=1,
        n_modes=(16,),
        size=(128,),
        hidden_channels=64,
        n_layers=4,
        positional_embedding="grid",
        non_linearity=F.relu,
        padding=0,
        dropout=0.0,
        norm=None
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
    
    print(f"Starting AW-FNO 1D training (Sobolev H1 Loss) on Burgers for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            out = model(batch_x)
            
            # Train using H1 Sobolev Loss
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
    plt.title('AW-FNO Burgers H1 Training Loss')
    plt.legend()
    
    # Test error plot
    plt.subplot(1, 3, 2)
    plt.plot(test_rel_history, label='Test Rel L2', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2')
    plt.title('AW-FNO Burgers Test Performance')
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
    plt.savefig(os.path.join(results_dir, 'awfno_burgers_h1_loss_plot.png'))
    print(f"Results plot saved to {os.path.join(results_dir, 'awfno_burgers_h1_loss_plot.png')}")
    
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
        plt.plot(pred_y, '--', label='AW-FNO-H1 Prediction', color='red', linewidth=2)
        plt.title(f'AW-FNO Burgers 1D (H1 Loss): GT vs Prediction\nTest Rel L2: {test_rel_history[-1]:.6f}')
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
        field_plot_path = os.path.join(results_dir, 'awfno_burgers_h1_field_comparison.png')
        plt.savefig(field_plot_path)
        print(f"Field comparison with error plot saved to {field_plot_path}")

if __name__ == "__main__":
    train_burgers_awfno_h1()
