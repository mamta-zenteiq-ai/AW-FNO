# analyse the data /media/HDD/mamta_backup/datasets/PDEBench/shallow_water/2D/shallow-water/2D_rdb_NA_NA.h5

import h5py
import numpy as np  
import matplotlib.pyplot as plt  

# open the file in read mode
file_path = '/media/HDD/mamta_backup/datasets/PDEBench/shallow_water/2D/shallow-water/2D_rdb_NA_NA.h5'
hf = h5py.File(file_path, 'r') 

# Get all sample keys (e.g., '0000', '0001', ...)
sample_keys = sorted(list(hf.keys()))
print(f"Total samples in the file: {len(sample_keys)}")
print(f"First 5 sample keys: {sample_keys[:1]}")

# Load the first sample
first_sample = hf[sample_keys[3]]
data = np.array(first_sample['data'])
grid_t = np.array(first_sample['grid/t'])
grid_x = np.array(first_sample['grid/x'])
grid_y = np.array(first_sample['grid/y'])

print("\nFirst sample ('0000') details:")
print(f"data shape (t, x, y, variable): {data.shape}")
print(f"grid_t shape: {grid_t.shape}")
print(f"grid_x shape: {grid_x.shape}")
print(f"grid_y shape: {grid_y.shape}")

# print dimension of variable
print(f"variable dimension: {data.shape[-1]}")

print("Full shape:", data.shape)
print("Number of variables:", data.shape[-1])

for i in range(data.shape[-1]):
    print(f"Channel {i}:")
    print("  min =", data[..., i].min())
    print("  max =", data[..., i].max())

# # print last time step data of variable
# print(f"{data[-1, 0, 0, 0]:.6f}")

# Convert dataset to numpy array for the first channel
sample_data = data[..., 0] # shape (t, x, y)
vmin = sample_data.min()
vmax = sample_data.max()

print(f"\nData range: min = {vmin:.4f}, max = {vmax:.4f}")

# Plot multiple timesteps to see the evolution with a constant scale
timesteps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
fig, axes = plt.subplots(2, 6, figsize=(18, 7))  # Adjusted aspect ratio for 2x6
axes = axes.flatten()

# Keep track of active axes for the colorbar
active_axes = []

for i, t in enumerate(timesteps):
    ax = axes[i]
    im = ax.imshow(sample_data[t], extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]], 
                   origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
    ax.set_title(f"t = {grid_t[t]:.2f} (step {t})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    active_axes.append(ax)

# Hide any unused subplots
for j in range(len(timesteps), len(axes)):
    fig.delaxes(axes[j])

# Add a shared colorbar using only the active axes
fig.colorbar(im, ax=active_axes, label='Water Height')
plt.suptitle("Shallow Water Equation 2D Evolution (Constant Scale)", fontsize=16)
plt.savefig("swe_2d_evolution.png", bbox_inches='tight')
print("\nEvolution plot saved to swe_2d_evolution.png")