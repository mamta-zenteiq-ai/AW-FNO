import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from awfno.models.fno import FNO
from awfno.models.awfno_v2 import AWFNOv2_3d
from awfno.utils.unit_gaussian_normalization import UnitGaussianNormalizer

def visualize_comparison():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    data_path = '/media/HDD/mamta_backup/datasets/fno/navier_stokes'
    test_data  = torch.load(os.path.join(data_path, 'ns_test_64.pt'))
    x_test  = test_data['x'].float()
    y_test  = test_data['y'].float()
    
    # Reshape to 5D for 3D models: [B, C, T, H, W]
    x_input = x_test.unsqueeze(1).unsqueeze(2) 
    y_ground = y_test.unsqueeze(1).unsqueeze(2)
    
    # Normalize
    x_normalizer = UnitGaussianNormalizer(x_input)
    x_encoded = x_normalizer.encode(x_input).to(device)
    
    y_normalizer = UnitGaussianNormalizer(y_ground)
    y_normalizer.to(device)

    # 2. Load FNO Model
    fno_model = FNO(
        n_modes=(1, 12, 12),
        in_channels=1,
        out_channels=1,
        hidden_channels=128,
        n_layers=4,
        positional_embedding="grid",
        use_channel_mlp=False
    ).to(device)
    
    fno_path = os.path.join(PROJECT_ROOT, 'results', 'fno_ns', 'fno_ns_best.pt')
    if os.path.exists(fno_path):
        fno_model.load_state_dict(torch.load(fno_path, map_location=device))
        print("Loaded FNO weights.")
    else:
        print("FNO weights not found, using initialized model.")

    # 3. Load AW-FNO v2 Model
    awfno_model = AWFNOv2_3d(
        in_channels=1,
        out_channels=1,
        n_modes=(1, 12, 12),
        size=(1, 64, 64),
        hidden_channels=16,
        n_fno_layers=4,
        n_wno_layers=4,
        wno_wavelet='db6'
    ).to(device)
    
    awfno_path = os.path.join(PROJECT_ROOT, 'results', 'awfno_v2_ns', 'awfno_v2_ns_best.pt')
    if os.path.exists(awfno_path):
        awfno_model.load_state_dict(torch.load(awfno_path, map_location=device))
        print("Loaded AW-FNO weights.")
    else:
        print("AW-FNO weights not found, using initialized model.")

    # 4. Inference
    fno_model.eval()
    awfno_model.eval()
    
    sample_idx = 0
    with torch.no_grad():
        fno_out = fno_model(x_encoded[sample_idx:sample_idx+1])
        awfno_out = awfno_model(x_encoded[sample_idx:sample_idx+1])
        
        fno_out = y_normalizer.decode(fno_out).cpu().numpy()[0, 0, 0]
        awfno_out = y_normalizer.decode(awfno_out).cpu().numpy()[0, 0, 0]
        ground_truth = y_test[sample_idx].numpy()

    # 5. Plotting (Publication Quality)
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Settle the scale: unify vmin and vmax across all plots
    v_min = min(ground_truth.min(), fno_out.min(), awfno_out.min())
    v_max = max(ground_truth.max(), fno_out.max(), awfno_out.max())
    
    cmap = 'viridis'
    
    # Ground Truth
    im = axes[0].imshow(ground_truth, cmap=cmap, vmin=v_min, vmax=v_max)
    # axes[0].set_title("Ground Truth", fontweight='bold')
    
    # FNO Prediction
    axes[1].imshow(fno_out, cmap=cmap, vmin=v_min, vmax=v_max)
    # axes[1].set_title("FNO 3D Prediction", fontweight='bold')
    
    # AW-FNO Prediction
    axes[2].imshow(awfno_out, cmap=cmap, vmin=v_min, vmax=v_max)
    # axes[2].set_title("AW-FNO v2 3D Prediction", fontweight='bold')
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.012, 0.7]) # Thinner width: 0.012
    fig.colorbar(im, cax=cbar_ax)

    output_path = os.path.join(PROJECT_ROOT, 'results', 'ns_comparison_publication.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot (thinner color bar) saved to {output_path}")

if __name__ == "__main__":
    visualize_comparison()
