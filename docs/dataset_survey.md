# Dataset Survey for AW-FNO Benchmarking

## Primary Datasets (Used in Paper)

### 1. Navier-Stokes 2D — FNO Benchmark (PRIMARY)

| Property | Value |
|---|---|
| Source | Li et al. FNO (2021), hosted by original authors |
| Download | `python datasets/download_fno_data.py --dataset ns2d` |
| Mirror | https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt- |
| Format | `.pt` (PyTorch tensors) — dict with keys `x`, `y` |
| Train | `ns_V1e-3_N1000_T50.pt` → 1000 trajectories, T=50 steps |
| Alt (64²) | `ns_train_64.pt` / `ns_test_64.pt` (project-local pre-split format) |
| Resolution | 64 × 64 spatial, 50 timesteps |
| Input | Vorticity ω at t=1..10, shape `(N, 64, 64, 10)` → `(N, 10, 64, 64)` |
| Output | Vorticity ω at t=11..20 (or t=40), shape `(N, 64, 64, 10)` |
| Re | 1000 |
| Size | ~300 MB |
| License | Research use (contact authors for redistribution) |
| Preprocessing | Unit Gaussian normalisation per field |
| Split | 1000 train / 200 test (standard) |
| Why | The canonical FNO/WNO benchmark — all baselines are reported on this |

**Standard results (published):**

| Model | Rel L2 @ t=10 |
|---|---|
| FNO (Li et al. 2021) | 0.0086 |
| WNO (Tripura et al. 2022) | ~0.012 |
| *AW-FNO (ours, target)* | *< 0.010* |

### 2. Burgers 1D — FNO Benchmark (SECONDARY)

| Property | Value |
|---|---|
| Source | Li et al. FNO (2021) |
| Download | `python datasets/download_fno_data.py --dataset burgers1d` |
| Format | MATLAB `.mat` — variables `a` (input), `u` (output) |
| Train | 1000 samples |
| Test | 200 samples |
| Resolution | 8192 points downsampled to 1024 |
| Input | Initial condition u(x, 0), shape `(N, 1024)` |
| Output | Solution u(x, 1), shape `(N, 1024)` |
| Size | ~50 MB |
| License | Research use |
| Why | Classic 1D shock-forming PDE; shows model handles discontinuities |

---

## Recommended Additional Datasets

### 3. Darcy Flow 2D (Triangular mesh)

| Property | Value |
|---|---|
| Source | Li et al. FNO (2021) |
| Format | `.mat` |
| Task | Coefficient-to-solution: a(x) → u(x) |
| Resolution | 421 × 421 downsampled to 85 × 85 or 141 × 141 |
| Why | Steady-state PDE; tests operator generalisation beyond time-marching |
| Notes | Not turbulent; less relevant for this paper's turbulence framing |

### 4. PDEBench — Compressible NS 2D

| Property | Value |
|---|---|
| Source | Takamoto et al. (2022), https://github.com/pdebench/PDEBench |
| Download | https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986 |
| Format | HDF5 |
| Task | Forward simulation of 2D compressible Navier-Stokes |
| Fields | density ρ, velocity (u,v), pressure p |
| Resolution | 512 × 512 (full), subsampled to 128 × 128 |
| Size | ~80 GB (full), ~5 GB per configuration |
| License | CC-BY 4.0 |
| Why | Includes shocks and compressible effects; strongly motivates wavelet branch |
| Preprocessing | Per-field normalisation required (extreme value ranges) |

### 5. PDEBench — Burgers 1D + 2D

| Property | Value |
|---|---|
| Source | Takamoto et al. (2022) |
| Format | HDF5 |
| Resolution | 1024 (1D), 128×128 (2D) |
| Variants | Multiple viscosities ν ∈ {1e-2, 1e-3, 1e-4} |
| License | CC-BY 4.0 |
| Why | Harder than FNO Burgers; tests low-viscosity shock formation |

### 6. PDEArena — Navier-Stokes 2D (forced turbulence)

| Property | Value |
|---|---|
| Source | Gupta & Brandstetter (2023), https://microsoft.github.io/pdearena |
| Download | Hugging Face: `pdearena/NavierStokes-2D` |
| Format | NetCDF4 / HDF5 |
| Resolution | 64 × 64, multiple timesteps |
| Variants | Incompressible (Kolmogorov forcing), compressible |
| License | MIT |
| Why | Modern benchmark with train/val/test splits, multiple Re values |

### 7. TurbBench / Johns Hopkins Turbulence Database

| Property | Value |
|---|---|
| Source | https://turbulence.pha.jhu.edu |
| Format | Web API / direct download |
| Resolution | 1024³ (isotropic), 5376 × 2048 (channel flow) |
| Why | Ground truth DNS; use for high-Re evaluation |
| Notes | Requires registration; large storage (TB range) |

---

## Dataset Priority for This Paper

For ICCFD13 (short paper, 6 pages):

| Priority | Dataset | Reason |
|---|---|---|
| **1 — Required** | NS 2D 64×64 (FNO benchmark) | All baselines reported here |
| **2 — Recommended** | Burgers 1D (FNO benchmark) | Validates 1D shock handling |
| **3 — Optional** | PDEBench CompressibleNS | Strongly supports wavelet claim |
| **4 — Future work** | PDEArena NS | Multi-Re generalisation |

---

## Data Directory Layout

After running `datasets/download_fno_data.py`, data should be at:

```
data/
├── ns2d/
│   ├── ns_V1e-3_N1000_T50.pt       # Full dataset
│   ├── ns_train_64.pt               # Pre-split 64×64
│   └── ns_test_64.pt
├── burgers1d/
│   ├── burgers_data_R10.mat
│   └── burgers_R10_N1000.mat
└── darcy/
    ├── piececonst_r421_N1024_smooth1.mat
    └── piececonst_r421_N1024_smooth2.mat
```

---

## Preprocessing Standards

All datasets follow this contract before being fed to any model:

1. **Shape convention**: `(B, C, *spatial_dims)` — channels first.
2. **Normalisation**: `UnitGaussianNormalizer` — zero mean, unit variance computed on training
   set only. Applied to both input and target.
3. **Train/test split**: stratified random split with `seed=42`.
4. **Reproducibility**: all splits use a fixed `generator=torch.Generator().manual_seed(42)`.
5. **Dtype**: `float32` throughout (bfloat16 optional for AMP).
