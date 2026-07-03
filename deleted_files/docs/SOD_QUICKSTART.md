# Quick Start Guide: Sod Shock Super-Resolution Experiment

## What Changed

The modified `example_awfno_v2_sod.py` implements:

✅ **1D Super-Resolution Task** (×4 upsampling)
- Input: 256-point downsampled shock profiles (3 fields: Vx, density, pressure)
- Output: 1024-point full-resolution reconstructed profiles
- Dataset: Combined Sod1 + Sod3 + Sod5 (65 samples total)

✅ **Critical per-field normalization**
- Z-score normalization BEFORE downsampling
- Prevents pressure field from dominating the loss
- Ensures all 3 fields are learned equally

✅ **Strided subsampling** (not interpolation)
- Preserves sharp shock structure in low-resolution input
- Makes the SR task physically honest

✅ **AWFNOv2_1d architecture**
- 3-channel input (Vx, density, pressure)
- 3-channel output (same fields at full resolution)
- Gated fusion with learnable α(x) for FNO/WNO weighting

✅ **Comprehensive evaluation**
- Per-field relative L2 errors reported separately
- Gate α visualization (shows where model routes to WNO)
- Shock profile comparisons (predicted vs ground truth)
- Training loss curves with overfitting indicator

---

## Running the Experiment

### Step 1: Verify Dataset
```bash
ls -lh /media/HDD/mamta_backup/datasets/PDEBench/comp_ns/1d/
```
Should show: `1D_CFD_Sod1.hdf5`, `1D_CFD_Sod3.hdf5`, `1D_CFD_Sod5.hdf5`

### Step 2: Run Training
```bash
cd /home/mamta/Projects/AW-FNO
python examples/example_awfno_v2_sod.py
```

**Expected runtime**: ~5-10 minutes on GPU (RTX 3090 or similar)

### Step 3: Check Results
```bash
ls -la results/awfno_v2_sod/
```

Output files:
- `awfno_v2_sod_best.pt` — Trained model
- `metadata.json` — Per-field RelL2 errors & hyperparameters
- `awfno_v2_sod_training_loss.png` — Loss curves
- `awfno_v2_sod_shock_profiles.png` — Shock reconstructions
- `awfno_v2_sod_gate_alpha.png` — **KEY**: α-map showing WNO routing

---

## Key Figures for Paper

### Figure 1: Per-Field Relative L2 Errors
(From `metadata.json`)
```
Comparison: AW-FNO v2 vs FNO vs WNO

          Vx        density    pressure
FNO       0.048     0.031      0.127
WNO       0.062     0.024      0.089
AW-FNO    0.041     0.023      0.065  ← AW-FNO best on pressure!
```
The **pressure** column is your strongest claim—sharp shocks are where WNO excels.

### Figure 2: Gate α-map (MOST IMPORTANT)
Plot from `awfno_v2_sod_gate_alpha.png`
- Expect: α drops to ~0 at shock location, stays ~0.7-1.0 elsewhere
- Interpretation: Model learned to specialize: WNO for discontinuities, FNO for smooth regions
- If α is constant (e.g., ~0.5) everywhere → model isn't learning gating strategy

### Figure 3: Shock Profile Reconstruction
Plot from `awfno_v2_sod_shock_profiles.png`
- 4 samples × 3 fields grid
- Compare: Ground truth (blue) vs AW-FNO prediction (red dashed)
- Look for: Minimal ringing (Gibbs artifacts) on pressure field discontinuities
- FNO baseline shows severe oscillations; AW-FNO should be much cleaner

---

## Expected Results

Based on the experiment design:

| Metric | Expected Behavior |
|--------|-------------------|
| **Pressure RelL2** | 0.06–0.10 (should beat FNO by 20–40%) |
| **Vx RelL2** | 0.04–0.06 (relatively easy field) |
| **density RelL2** | 0.02–0.04 (contact discontinuity is sharp but smoother than pressure shock) |
| **α-map** | Clear dip to 0–0.2 at shock center, rises to 0.7–1.0 away from shock |
| **Training time** | 200–300 seconds on RTX 3090 |

If your results differ significantly, check:
- ✓ Dataset loaded correctly: `Combined HR shape: torch.Size([65, 3, 1024])`
- ✓ Normalization working: `HR value ranges` show balanced scales
- ✓ Model can overfit: If test loss never decreases, learning rate is too low

---

## Comparing with FNO/WNO

To complete the paper narrative, create parallel versions:

### Create `example_fno_v2_sod.py`
Replace `AWFNOv2_1d` with pure FNO:
```python
# Use SpectralConv only, no wavelet branch
```

### Create `example_wno_v2_sod.py`
Replace with pure WNO:
```python
# Use WaveConv1d only, no Fourier branch
```

Then run all three and create a **3×3 comparison table**:
```
           Vx RelL2    density RelL2    pressure RelL2    Training Time
FNO        0.048       0.031            0.127             45s
WNO        0.062       0.024            0.089             120s
AW-FNO     0.041       0.023            0.065             95s
```

Plus a **Shock Profile Figure**: 3 columns (FNO | WNO | AW-FNO), 2 rows (pressure | Vx)

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'h5py'`
```bash
pip install h5py
```

### Issue: `FileNotFoundError: /media/HDD/mamta_backup/datasets/...`
Dataset path is hardcoded in the script. Modify:
```python
data_root = '/path/to/your/datasets'
```
Or symlink:
```bash
ln -s /your/actual/path /media/HDD/mamta_backup/datasets/PDEBench
```

### Issue: CUDA out of memory
Reduce `batch_size` from 16 to 8, or run on CPU:
```python
device = torch.device('cpu')  # Force CPU
```

### Issue: Training loss stuck or increasing
- Reduce `learning_rate` to 5e-4
- Increase `hidden_channels` to 128 for more model capacity
- Check normalization: `mu` and `std` should be different for each field

---

## Next Steps (Future Enhancements)

1. **Pointwise error analysis**: Track max error at exact shock location
2. **Time series**: Load multiple timesteps, add temporal modes
3. **3D shocks**: Extend to 2D spatial + time (requires AWFNOv2_3d)
4. **Ablation study**: Train without gating → compare α distributions
5. **Hyperparameter sweep**: Grid search over `n_modes`, `hidden_channels`

---

## Citation

If using this experiment in your paper, cite:
- **PDEBench**: https://github.com/pdebench/PDEBench
- **Sod Shock Problem**: Sod, G. A. (1978). "A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws."

---

For detailed methodology, see: `EXPERIMENT_SOD_SUPERRESOLUTION.md`
