# AW-FNO v2 on 1D Sod Shock Super-Resolution

## Overview

This document describes the modified experiment for AW-FNO v2 on compressible shock reconstruction using the PDEBench Sod shock datasets.

### Task Definition
**Spatial Super-Resolution (×4)**: Reconstruct 1024-point sharp shock profiles from 256-point downsampled versions.

Unlike time-evolution problems, this task tests the core claim of AW-FNO: that the model can recover sharp discontinuities better than pure Fourier methods by routing through the wavelet branch near shocks.

---

## Dataset Specification

### Files Used
- `1D_CFD_Sod1.hdf5` (41 samples)
- `1D_CFD_Sod3.hdf5` (12 samples)  
- `1D_CFD_Sod5.hdf5` (12 samples)

**Total: 65 samples** → 80/20 split (52 train, 13 test)

Location: `/media/HDD/mamta_backup/datasets/PDEBench/comp_ns/1d/`

### Data Structure
Each HDF5 file contains:
```
Vx:        (N_samples, 1024)  — velocity
density:   (N_samples, 1024)  — density
pressure:  (N_samples, 1024)  — pressure
```

---

## Input/Output Design

### Input: Low-Resolution (LR)
- **Shape**: `(N_samples, 3, 256)`
- **Creation**: Strided subsampling by ×4 (every 4th point)
  ```python
  lr = hr[:, :, ::4]  # NOT interpolation — preserves shocks
  ```
- **Channels**: `[Vx, density, pressure]`

### Output: High-Resolution (HR)
- **Shape**: `(N_samples, 3, 1024)`
- **Content**: Full-resolution ground truth fields

### Normalization (Critical)
Per-field z-score normalization **BEFORE downsampling**:
```python
mu  = x.mean(dim=(0, 2), keepdim=True)      # mean per field
std = x.std(dim=(0, 2), keepdim=True)       # std per field
x_norm = (x - mu) / (std + 1e-8)
```

**Why critical**: Sod pressure range (0–1000) is ~55× larger than Vx range (0–18.69). Without normalization, loss is dominated by pressure and the model ignores Vx/density.

---

## Model Architecture

### AWFNOv2_1d Configuration
```python
AWFNOv2_1d(
    in_channels=3,              # [Vx, density, pressure]
    out_channels=3,
    n_modes=64,                 # Fourier modes for 1D
    size=[256],                 # LR spatial resolution
    hidden_channels=64,
    n_fno_layers=4,             # Fourier layers
    n_wno_layers=4,             # Wavelet layers
    wno_wavelet='db6',          # Daubechies-6 wavelets
    padding=0,
    dropout=0.0,
)
```

**Key design**: The gated fusion layer learns **α(x)** — a spatially-dependent weight controlling the blend:
- **α → 1 (FNO dominant)**: Away from shocks (smooth regions)
- **α → 0 (WNO dominant)**: At shocks (discontinuities)

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Epochs** | 500 |
| **Batch Size** | 16 |
| **Learning Rate** | 1e-3 |
| **Optimizer** | Adam (weight decay 1e-4) |
| **Scheduler** | StepLR (step=100, gamma=0.5) |
| **Loss** | L2 (per-batch MSE) |

---

## Evaluation Metrics

### 1. Per-Field Relative L2 Error (Primary Metric)
Computed separately for each field on the test set:

$$\text{RelL2}_{field} = \frac{\|\hat{y}_{field} - y_{field}\|_2}{\|y_{field}\|_2}$$

Reported independently for:
- **Vx** (velocity)
- **density** (ρ)
- **pressure** (P)

**Interpretation**: Pressure L2 near the shock is the hardest test. Lower AW-FNO pressure L2 than FNO indicates successful wavelet routing to shocks.

### 2. Gate Visualization (α-map)
Plot α(x) as a function of spatial position (LR domain, 256 pts):
- If α dips sharply to ~0 at the shock location → model learned correct routing
- Smooth, constant α → model failed to specialize

### 3. Shock Profile Visualization
Predicted vs. ground truth profiles, zoomed at discontinuity:
- **FNO baseline** exhibits Gibbs oscillations (ringing artifacts)
- **AW-FNO** should show sharp reconstruction with minimal ringing

### 4. Max Pointwise Error at Shock Front
Track the maximum reconstruction error at the discontinuity location (future enhancement).

---

## Output Files

All results saved to: `results/awfno_v2_sod/`

| File | Description |
|------|-------------|
| `awfno_v2_sod_best.pt` | Trained model weights |
| `awfno_v2_sod_training_loss.png` | 4-panel loss plot: overall L2, per-field train/test, overfitting ratio |
| `awfno_v2_sod_shock_profiles.png` | 4 samples × 3 fields: predicted vs ground truth profiles |
| `awfno_v2_sod_gate_alpha.png` | α-map: gate visualization at shock location |
| `metadata.json` | Training metadata and per-field RelL2 errors |

---

## Running the Experiment

### Quick Start
```bash
cd /home/mamta/Projects/AW-FNO
python examples/example_awfno_v2_sod.py
```

### Expected Output
```
Using device: cuda
Loading Sod shock datasets...
Loading 1D_CFD_Sod1.hdf5...
  Loaded (41, 3, 1024)
Loading 1D_CFD_Sod3.hdf5...
  Loaded (12, 3, 1024)
Loading 1D_CFD_Sod5.hdf5...
  Loaded (12, 3, 1024)
Combined HR shape: torch.Size([65, 3, 1024])
...
Training completed in XXs
========================================================
PER-FIELD RELATIVE L2 ERRORS (NORMALIZED)
========================================================
Vx          : 0.XXX
density     : 0.XXX
pressure    : 0.XXX
========================================================
```

---

## Key Implementation Details

### 1. Strided Subsampling (Not Interpolation)
```python
lr = hr[:, :, ::downsample_factor]  # Preserves sharp shock structure
```
This is **crucial**: interpolation smooths the shock, making the SR task artificially easy and physically dishonest.

### 2. Gate Capture
The gate α is extracted after the Fourier and Wavelet branches converge:
```python
alpha = model.gate.sigmoid(model.gate.gate_conv(torch.cat([v_f, v_w], dim=1)))
```
Averaged over hidden channels to produce a 1D map: α(x).

### 3. Per-Field Loss Tracking
During training, we monitor 4 loss curves:
- **Overall L2**: standard training metric
- **Train L2 per field**: sanity check (should all decrease)
- **Test L2 per field**: field-specific generalization
- **Test/Train ratio**: overfitting detector (>1 = overfitting)

---

## Physical Interpretation

### Sod Shock Structure
At the shock front:
- **Vx**: Fan → plateau (smooth transition)
- **density (ρ)**: Contact discontinuity (sharp jump)
- **pressure (P)**: Shock front (very steep gradient)

### Why AW-FNO Should Succeed
1. **Fourier (FNO alone)**: Global basis → oscillations near discontinuities (Gibbs phenomenon)
2. **Wavelets (WNO alone)**: Localized basis → sharp reconstruction, but less accurate in smooth regions
3. **AW-FNO gated fusion**: 
   - α ≈ 1 in smooth regions (FNO dominates)
   - α ≈ 0 at shocks (WNO dominates)
   - Best of both worlds

---

## Reproducibility

- **Random seed**: 42
- **CUDA determinism**: Enabled
- **Dataset**: Fixed order (Sod1 + Sod3 + Sod5)
- **Train/test split**: 80/20 fixed random permutation (seeded)

---

## Hyperparameter Tuning Notes

If results are unsatisfactory, consider:
- **Batch size**: Increase to 32 if memory allows (smoother gradient)
- **Learning rate**: Reduce to 5e-4 if training diverges
- **Hidden channels**: Increase from 64 to 128 for more capacity
- **Epochs**: Extend to 1000 if validation continues improving
- **Wavelet**: Try `db4` (smoother) or `db8` (sharper) instead of `db6`

---

## Comparison with FNO/WNO

For a complete paper result, run parallel experiments:
- `example_fno_burgers.py` (modified for Sod)
- `example_wno_burgers.py` (modified for Sod)

Then compare the three per-field RelL2 tables and shock profile plots. The gap in pressure L2 and α visualization will be your key figures.

---

## References

- **PDEBench**: https://github.com/pdebench/PDEBench
- **AW-FNO v2**: Gated wavelet-Fourier fusion (this project)
- **Sod shock**: Riemann solver benchmark for compressible Euler equations
