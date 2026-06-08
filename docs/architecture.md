# AW-FNO Architecture Details

## v1 — Per-Layer GFM (Paper Primary)

Each AW-FNO block runs a local Fourier + wavelet convolution **in parallel** and fuses
results with an adaptive gate.  The gate is re-computed at every block, allowing the
model to switch representations layer-by-layer.

```
AWFNOBlock2d
─────────────────────────────────────────
Input x  ─────────────────────────────┐
         │                            │
         ├──► SpectralConv2d          │
         │     (FFT → R_φ · · → IFFT) │
         │     → V_fno               │
         │                            │
         ├──► WaveConv2d              │
         │     (DWT → R_ψ · · → IDWT) │
         │     → V_wno               │
         │                            │
         ├──► AdaptiveGatedFusion2d   │
         │     cat = [V_fno, V_wno]   │
         │     α   = σ(Conv1×1(cat))  │
         │     → α⊙V_fno+(1−α)⊙V_wno │
         │                            │
         └──► skip_connection ────────┤
                                      │
                        [add] ◄───────┘
                          │
                     LayerNorm
                          │
                        GELU
                          │
                       Output
```

**Global architecture:**
```
Input
  └─ GridEmbedding2D  (append x,y grid coords)
  └─ Lifting ChannelMLP
  └─ AWFNOBlock2d × n_layers
  └─ Projection ChannelMLP
  └─ Output
```

## v2 — Branch-Parallel (Variant)

v2 runs two completely separate deep networks then fuses them once:

```
Input
  └─ GridEmbedding2D
  └─ Lifting

  ┌──────────────────────────┐  ┌──────────────────────────┐
  │  FourierBranch           │  │  WaveletBranch           │
  │  [SpectralConv + skip] × n │  │  [WaveConv + skip] × n   │
  └──────────────────────────┘  └──────────────────────────┘
               │                              │
               └───────── GatedFusion ────────┘
                          InstanceNorm
               │
         + lifting residual
               │
          Projection
               │
            Output
```

v2 allows each branch to develop specialised representations before merging.
The trade-off is higher memory usage (both branches full-depth simultaneously).

## Gate Initialisation

Both versions initialise the gate convolution weights to zero:
```python
nn.init.constant_(self.gate_conv.weight, 0)
nn.init.constant_(self.gate_conv.bias, 0)
```
This means α = σ(0) = 0.5 at epoch 0, giving each branch equal weight at the
start of training.  The gate then adapts based on gradient signal — regions with
sharp gradients push α toward 0 (WNO dominance) and smooth regions toward 1.

## SpectralConv2d

Implements the FNO integral kernel:
```
K_F(v)(x) = F⁻¹(R_φ · F(v))(x)
```
Only the first `n_modes_x × n_modes_y` Fourier modes are retained, acting as
a learned low-pass filter in frequency space.

Implementation uses `tensorly` / `tltorch` for optional tensor factorization
(Tucker, CP) to reduce parameter count.

## WaveConv2d

Implements the WNO integral kernel:
```
K_W(v)(x) = W⁻¹(R_ψ · W(v))(x)
```
Uses `pytorch_wavelets.DWT` / `IDWT` with `db6` wavelet (default) and
`periodic` boundary mode (consistent with FNO's periodicity assumption).

Learnable weights are placed on:
- the approximation subband (low-frequency)
- all detail subbands at each decomposition level

## Complexity Analysis

| Component | FLOPs | Parameters |
|---|---|---|
| SpectralConv2d (n_modes=12, C=32) | O(N log N + n_modes² C²) | 2 × n_modes² × C² |
| WaveConv2d (level=3, db6, C=32) | O(N C) | ≈ 3 × 4 × (N/8)² × C² |
| AdaptiveGatedFusion2d (C=32) | O(N C²) | 2C × C = 2C² |
| AWFNOBlock2d total | — | ≈ SpectralConv + WaveConv + GFM + skip |

For the paper configuration (C=32, n_modes=12, H=W=64, n_layers=4):
- Total params: ~7–9 M (comparable to FNO at same width)

## Normalisation

Uses `LayerNorm` (default) applied over the channel dimension:
- Input: (B, H, W, C) after spatial permute
- Normalise over C
- Permute back to (B, C, H, W)

`InstanceNorm2d` is used in v2's GatedFusion (normalises per-sample, per-channel).
