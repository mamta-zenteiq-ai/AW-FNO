# AW-FNO: Research Overview

## Scientific Problem

### Governing Physics
We address the **temporal super-resolution of 2D incompressible turbulent flows** governed by the
Navier–Stokes equations (vorticity-streamfunction formulation):

```
∂ω/∂t + u·∇ω = ν∇²ω + f
```

where `ω` is the vorticity scalar, `u` the velocity field, `ν = 1/Re` the kinematic viscosity, and
`f` an optional forcing term. At Re = 1000 the flow is turbulent: kinetic energy cascades from
large coherent eddies down to dissipative small-scale structures, creating a wide range of active
spatial frequencies.

We also validate on the **1D viscous Burgers equation** as a controlled benchmark with shock
formation:

```
∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
```

### Task Formulation
**Supervised operator learning**: given low-resolution or incomplete observations of a PDE solution
at time `t`, predict the high-resolution solution at time `t + Δt`.

| Property | Value |
|---|---|
| Input | Vorticity field ω at 10 past timesteps, shape `(B, 1, 64, 64)` |
| Output | Vorticity field ω at 10 future timesteps, shape `(B, 1, 64, 64)` |
| Resolution | 64 × 64 (spatial), trained on a single resolution |
| Prediction type | Direct (non-autoregressive per step, autoregressive over windows) |
| Physics domain | Doubly periodic 2D box, Re = 1000 |
| Metric | Relative L2 error (`‖pred − gt‖₂ / ‖gt‖₂`) |

The "super-resolution" claim in the paper refers to two aspects:
1. The model must reconstruct fine-scale turbulent features not explicitly encoded in the sparse
   temporal input (temporal SR).
2. Zero-shot spatial SR: trained at 64×64 and evaluated at 128×128 via resolution-invariant
   spectral/wavelet operators.

---

## AW-FNO Contribution

### Core Idea
Standard FNO operates globally in the Fourier domain and cannot localise energy near sharp
features — it suffers from Gibbs oscillations near vortex cores and shear layers. WNO operates
locally via compactly supported wavelets and captures sharp gradients but misses long-range
correlations. AW-FNO fuses both via a **spatially adaptive gating mechanism** that learns, at each
spatial location, whether to trust the FNO branch (smooth regions) or the WNO branch (sharp
gradients).

### Architecture (v1 — paper-primary)
```
Input (B, C_in, H, W)
  ↓  GridEmbedding2D: append (x, y) grid coords → (B, C_in+2, H, W)
  ↓  Lifting MLP: point-wise → (B, C_hid, H, W)
  ↓  [AWFNOBlock2d] × n_layers:
     │  ├─ SpectralConv2d (FFT → truncate → IFFT) → V_fno
     │  ├─ WaveConv2d (DWT → filter → IDWT)       → V_wno
     │  ├─ AdaptiveGatedFusion2d:
     │  │    α = σ(Conv1×1([V_fno, V_wno]))       ← per-channel gate
     │  │    V_fused = α⊙V_fno + (1−α)⊙V_wno
     │  └─ skip + LayerNorm + GELU
  ↓  Projection MLP: point-wise → (B, C_out, H, W)
```

**Key properties:**
- The gate α ∈ (0, 1) is **spatial** and **per-channel** — different channels can specialise.
- Initialised to zero weights → starts at α = 0.5 (equal mix), adapts during training.
- The gate map is **interpretable**: after training, high-α regions indicate FNO dominance
  (smooth flow) and low-α regions indicate WNO dominance (vortex cores, shear layers).

### Architecture (v2 — branch-parallel variant)
```
Input → Lifting → ┌─ FourierBranch (n_fno_layers deep) ─┐
                  │                                      ├─ GatedFusion → Projection
                  └─ WaveletBranch (n_wno_layers deep) ──┘
```
v2 runs branches in parallel across all layers then merges once. This allows each branch to
develop specialised representations independently.

---

## Current Repository State

### What exists and works
| Component | Status |
|---|---|
| `awfno/models/awfno.py` — AWFNO1d + AWFNO2d (v1) | ✅ Implemented |
| `awfno/models/awfno_v2.py` — AWFNOv2_1d/2d (v2) | ✅ Implemented |
| `awfno/models/fno.py` — FNO (full-featured) | ✅ Implemented |
| `awfno/models/wno.py` — WNO1d/2d/3d | ✅ Implemented |
| `awfno/layers/` — spectral/wavelet convolutions | ✅ Implemented |
| Per-script training (NS + Burgers) | ✅ Working |
| Partial NS training results (100/500 epochs) | ✅ Partial |

### Missing pieces for publication
| Gap | Impact |
|---|---|
| No unified training script with configs | Blocks reproducibility |
| No gate α visualisation utility | Blocks interpretability claim |
| No ablation (no-gate variant) | Blocks proving GFM contribution |
| No dataset download script | Blocks external replication |
| No zero-shot SR evaluation | Blocks "zero-shot SR" claim in conclusions |
| Hardcoded data paths throughout | Blocks portability |
| No AMP / gradient clipping | Risk of instability on long runs |
| No checkpoint `best_model` saving for all runs | Blocks post-hoc evaluation |
| H1 norm loss not integrated in main pipeline | Misses physics-informed training option |
| No spectral energy comparison plot | Misses key turbulence analysis figure |
| Single test sample for visuals | Insufficient for paper |

---

## Evaluation Metrics

| Metric | Formula | Notes |
|---|---|---|
| Relative L2 | `‖pred−gt‖₂ / ‖gt‖₂` | Primary metric, per-sample then mean |
| MSE | `mean((pred−gt)²)` | Absolute scale |
| MAE | `mean(|pred−gt|)` | Robust to outliers |
| Spectral L2 | L2 in Fourier space | Captures frequency-specific errors |
| Max pointwise error | `max|pred−gt|` | Sensitive to Gibbs oscillations |
| Params (M) | Model size | Fairness of comparison |
| Training time (s/epoch) | Wall-clock | Practical efficiency |

---

## Minimum Experiments for Paper Acceptance

1. **Main table** — NS 64×64: AW-FNO vs FNO vs WNO, 500 epochs, seed 42
2. **Gate analysis figure** — learned α maps on ≥3 test samples
3. **Ablation** — AW-FNO-NoGate (α fixed = 0.5): proves gate contributes
4. **Visual comparison** — GT / FNO / WNO / AW-FNO + error maps, publication quality
5. **Convergence curves** — all three models on same axes
6. *(Recommended)* **Burgers 1D table** — secondary benchmark showing 1D generalisation
7. *(Recommended)* **Zero-shot SR** — train 64×64, eval 128×128 without retraining

---

## Open Scientific Questions

- Does the gate learn physically interpretable patterns, or is it a trivial averaging?
  → Addressed by gate visualisation + correlation with flow features
- How sensitive is performance to wavelet choice (db4 vs db6 vs Haar)?
  → Could be an ablation table row
- Does the GFM benefit scale with Re?
  → Future work: test at Re = 500, 1000, 5000
- Is v1 (per-layer) or v2 (branch-parallel) better?
  → Include both in ablation or present best in paper
