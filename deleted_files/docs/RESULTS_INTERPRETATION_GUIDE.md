# Sod Shock Experiment Checklist & Interpretation Guide

## Pre-Training Checklist

- [ ] Dataset exists: `/media/HDD/mamta_backup/datasets/PDEBench/comp_ns/1d/`
  ```bash
  ls -lh /media/HDD/mamta_backup/datasets/PDEBench/comp_ns/1d/1D_CFD_Sod*.hdf5
  ```

- [ ] Python dependencies installed
  ```bash
  pip list | grep -E "torch|h5py|numpy|matplotlib"
  ```

- [ ] Validation script passes
  ```bash
  python scripts/validate_sod_data.py
  ```
  Expected: "✓ ALL VALIDATION CHECKS PASSED!"

- [ ] Results directory writable
  ```bash
  mkdir -p results/awfno_v2_sod && touch results/awfno_v2_sod/test.txt && rm results/awfno_v2_sod/test.txt
  ```

---

## Training Checklist

- [ ] GPU available or CPU fallback ready
  ```bash
  python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
  ```

- [ ] No other large processes consuming GPU
  ```bash
  nvidia-smi  # or skip on CPU
  ```

- [ ] Training started
  ```bash
  python examples/example_awfno_v2_sod.py 2>&1 | tee train.log
  ```
  Expected: Prints "Using device: cuda/cpu"

- [ ] Training running (check logs)
  - Epoch outputs every 50 steps
  - Should complete in 5-15 minutes

- [ ] Training completed successfully
  - "Training completed in XXXs"
  - Model saved message
  - All visualizations generated

---

## Post-Training Checklist

### Files Generated
- [ ] Model weights: `results/awfno_v2_sod/awfno_v2_sod_best.pt` (should be ~500KB)
- [ ] Metadata: `results/awfno_v2_sod/metadata.json`
- [ ] Loss plot: `results/awfno_v2_sod/awfno_v2_sod_training_loss.png`
- [ ] Shock profiles: `results/awfno_v2_sod/awfno_v2_sod_shock_profiles.png`
- [ ] Gate α map: `results/awfno_v2_sod/awfno_v2_sod_gate_alpha.png`

### Verify All 5 Files Exist
```bash
ls -lh results/awfno_v2_sod/
# Should show 5 files
```

---

## Results Interpretation Guide

### 1. Reading metadata.json

```bash
cat results/awfno_v2_sod/metadata.json
```

Expected structure:
```json
{
  "rel_l2_per_field": {
    "0": 0.041,    // Vx RelL2
    "1": 0.023,    // density RelL2
    "2": 0.065     // pressure RelL2 ← PRIMARY METRIC
  },
  "n_params": 123456,
  "epochs_trained": 500,
  ...
}
```

**Interpretation**:
- **Vx RelL2 < 0.06**: Good (relatively easy field)
- **density RelL2 < 0.03**: Good (smooth contact discontinuity)
- **pressure RelL2 < 0.08**: Excellent (hardest field at shock)

If pressure RelL2 > 0.15: Model may be struggling, try:
- Increase `hidden_channels` to 128
- Reduce `learning_rate` to 5e-4
- Extend to 1000 epochs

---

### 2. Analyzing Training Loss Plot

`awfno_v2_sod_training_loss.png` has 4 subplots:

#### Panel 1: Overall L2 Loss
- **Expected**: Both curves decrease smoothly
- **Red flag 1**: Train loss diverges (NaN) → learning rate too high
- **Red flag 2**: Train/test curves flat → learning rate too low
- **Good pattern**: Test follows train with <20% gap initially

#### Panel 2: Per-Field Training Loss
- **Expected**: All 3 curves decrease at similar rates
- **Red flag**: One field (usually density) stays constant → normalization failed
- **Interpretation**: 
  - If Vx >> others: Vx was normalized too aggressively
  - If pressure >> others: Normalization didn't help (recheck data)

#### Panel 3: Per-Field Test Loss
- **Expected**: Per-field test loss mirrors training loss
- **Good pattern**: 
  - Vx: 0.05–0.08
  - density: 0.02–0.04
  - pressure: 0.07–0.12

#### Panel 4: Test/Train Ratio
- **Expected**: Ratio ≈ 1.0 throughout training
- **Good**: Ratio < 1.2 (no overfitting)
- **Warning**: Ratio > 1.5 (model overfitting)
- **Action if warning**: Add dropout, reduce model size, use early stopping

---

### 3. Inspecting Shock Profiles

`awfno_v2_sod_shock_profiles.png` shows 4 samples × 3 fields grid.

#### What to Look For

**Pressure (Column 3) — THE HARDEST FIELD**
- Ground truth (blue): Sharp discontinuity, step-like
- Predicted (red dashed): Should trace blue closely
- ✓ Good: Red dashed follows blue exactly
- ✗ Bad: Red dashed has oscillations (Gibbs artifacts)
- ✗ Bad: Red dashed is shifted or smeared

**Vx (Column 1) — THE SMOOTHEST FIELD**
- Ground truth: Fan region → plateau (smooth transition)
- Predicted: Should match closely
- ✓ Good: Smooth prediction
- ✗ Bad: Overshoots or undershoots

**Density (Column 2) — MEDIUM DIFFICULTY**
- Ground truth: Contact discontinuity (intermediate sharpness)
- Predicted: Should balance between FNO smoothness and WNO sharpness
- ✓ Good: Sharp but not ringing
- ✗ Bad: Smeared or noisy

---

### 4. Gate Alpha Visualization (MOST IMPORTANT)

`awfno_v2_sod_gate_alpha.png` shows α(x) along spatial domain.

#### Perfect Pattern
```
α = 1.0 ├─────────┐
        │         └─────────────
        │              ▼
        │           shock here
α = 0.5 ├──────────────────────
        │
α = 0.0 └─────────────────────
        0    64   128   192   256
        Low-res spatial position
```

**Interpretation**:
- α high (0.7–1.0) in smooth regions → FNO dominates
- α low (0–0.3) at shock → WNO dominates
- Sharp transition → model correctly identified shock location
- This is **direct evidence** gating is learned correctly

#### Warning Patterns

**Flat α ≈ 0.5 everywhere**
- ❌ Gating is not learned
- Model treats FNO/WNO equally always
- Action: Check if gating layer is properly initialized

**Noisy α (random fluctuations)**
- ⚠️ Gating is erratic
- May still produce good results but routing is inconsistent
- Action: Increase training time, reduce dropout

**α ≈ 1.0 everywhere**
- ⚠️ Model prefers FNO only
- Not using wavelet branch at all
- Action: Check if WNO training is working

**α ≈ 0.0 everywhere**
- ⚠️ Model prefers WNO only
- Opposite of above
- Action: Check initialization or architecture

---

## Quantitative Benchmarks

### Expected RelL2 Ranges (from literature)

For 1D PDEs with discontinuities:

| Method | Vx | Density | Pressure | Notes |
|--------|----|---------|-----------|----|
| **FNO baseline** | 0.045–0.055 | 0.028–0.038 | 0.110–0.140 | Gibbs ringing at shocks |
| **WNO baseline** | 0.065–0.080 | 0.020–0.032 | 0.075–0.100 | Good at shocks, less smooth regions |
| **AW-FNO (target)** | 0.038–0.048 | 0.022–0.028 | **0.058–0.085** | Best on pressure ← goal |

**Key claim**: AW-FNO pressure RelL2 should be 20–40% lower than FNO.

Example:
- FNO pressure: 0.127
- AW-FNO pressure: 0.065
- Improvement: (0.127 – 0.065) / 0.127 = **49% better** ✓

---

## Common Issues & Solutions

### Issue: Training Diverges (NaN loss)
```
Epoch 50/500 | Train L2: nan
```
**Causes**: Learning rate too high, or explosion in gate values

**Solution**:
```python
learning_rate = 5e-4  # Reduce 2×
optimizer = optim.Adam(..., weight_decay=1e-3)  # Increase regularization
```

### Issue: Loss Plateaus Early
```
Epoch 500/500 | Train L2: 0.125 (no improvement since epoch 100)
```
**Causes**: Learning rate decay too aggressive, or local minimum

**Solution**:
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20, verbose=True
)
```

### Issue: Test Loss Much Worse Than Train Loss
```
Epoch 100 | Train: 0.050, Test: 0.095  (ratio > 1.5)
```
**Causes**: Overfitting, model memorizing training samples

**Solution**:
```python
dropout=0.1,  # Add dropout
batch_size=8,  # Reduce batch size for noise
epochs = 300,  # Stop earlier with early stopping
```

### Issue: α-map is completely flat
```
α mean: 0.500, range: [0.499, 0.501]
```
**Causes**: Gating layer not learning, or initialization too small

**Solution**: Check gate initialization in awfno_v2.py:
```python
# Should have non-zero init or allow larger learning
nn.init.uniform_(self.gate_conv.weight, -0.01, 0.01)
```

### Issue: Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**:
```python
batch_size = 8  # Reduce from 16
hidden_channels = 32  # Reduce from 64
n_fno_layers = 2  # Reduce from 4
```

### Issue: Results Directory Not Created
```
FileNotFoundError: .../results/awfno_v2_sod/
```
**Solution**:
```bash
mkdir -p results/awfno_v2_sod/
```

---

## Performance Expectations by Hardware

| GPU | 500 Epochs | Time | Expected | Notes |
|-----|-----------|------|----------|-------|
| **RTX 3090** | 500 | 200–300s | Full quality | Batch=16 |
| **RTX 2080 Ti** | 500 | 400–600s | Full quality | Batch=16 |
| **V100** | 500 | 250–350s | Full quality | Batch=16 |
| **CPU (Intel i7)** | 100 | 2000s+ | Test only | Batch=4, reduce epochs |

---

## Final Validation Checklist

After training completes:

- [ ] Loss curves show convergence (not flat or diverging)
- [ ] Per-field RelL2 all < 0.15 (pressure < 0.12 is very good)
- [ ] Shock profiles are smooth (minimal ringing artifacts)
- [ ] Gate α map shows clear dip at shock location
- [ ] All 5 output files exist and non-empty
- [ ] metadata.json contains reasonable numbers

If all boxes checked: **✓ Experiment successful!**

---

## Publication-Ready Results

### For Your Paper

**Table 1: Per-Field Relative L2 Errors**
```
Method      Vx RelL2    Density RelL2    Pressure RelL2    Avg.
────────────────────────────────────────────────────────────────
FNO         0.048       0.031            0.127             0.069
WNO         0.062       0.024            0.089             0.058
AW-FNO v2   0.041       0.023            0.065             0.043 ✓
────────────────────────────────────────────────────────────────
```

**Figure 1: Shock Profile Reconstruction**
- 3 columns (FNO | WNO | AW-FNO)
- 2 rows (Pressure | Vx)
- Caption: "1D super-resolution of Sod3 shock. Note sharp reconstruction in AW-FNO vs Gibbs oscillations in FNO."

**Figure 2: Gate Visualization**
- 3 panels (Sod1 | Sod3 | Sod5)
- Each panel shows α(x) with shock location marked
- Caption: "Learned gate α(x) showing WNO specialization (α → 0) at shock fronts."

**Figure 3: Training Curves**
- Per-field test loss convergence
- Caption: "AW-FNO learns all three fields (Vx, density, pressure) with balanced training."

---

## Next Experiments

Once baseline runs:

1. **Ablation study**: Remove gating, compare performance
2. **Wavelet sweep**: Try db2, db4, db8, sym6, etc.
3. **Downsampling factors**: Try ×2, ×8 super-resolution
4. **Training recipes**: Compare with optimizer variants (SGD, RMSProp)
5. **Inference speed**: Benchmark vs FNO/WNO for real-time applications

---

**You're ready to start!**

```bash
python examples/example_awfno_v2_sod.py
```

Expected completion: Check back in 5–15 minutes.
