# AW-FNO: An Adaptive Approach for Fluid Flow Super-Resolution via Gated Wavelet-Fourier Learning

**ICCFD13-2026-311** · Thirteenth International Conference on Computational Fluid Dynamics, Milan, Italy, July 6–10, 2026

**Authors:** Mamta Saini\*, Parikshit Mahajan\*, Gazania Marine Jyrwa\*, Diya Nagchaudhury\* (\*Equal contribution)

**Affiliation:** Department of Computational and Data Sciences, Indian Institute of Science, Bengaluru, India

---

## Overview

**AW-FNO** (Augmented Wavelet Fourier Neural Operator) is a hybrid neural operator for super-resolution of turbulent fluid flows. It combines the Fourier Neural Operator (FNO) and the Wavelet Neural Operator (WNO) in a **parallel dual-branch architecture**, coupled through a learned **Adaptive Gated Fusion Mechanism (AGFM)**.

The AGFM generates a spatially varying gate that dynamically routes:
- the **FNO branch** in smooth, globally coherent regions
- the **WNO branch** near sharp gradients and local discontinuities

This design overcomes the spectral smoothing and Gibbs oscillations of standalone FNO and the large-scale coherence deficits of standalone WNO.

A practical property of AW-FNO is that disabling either branch through an architectural switch recovers the standalone behaviour of the other branch.

---

## Architecture

### 1. Fourier (FNO) Branch — Global spectral learning

Applies FFT, truncates to $k_{\max}$ low-frequency modes, applies learnable complex weights $R_\phi$, then applies inverse FFT:

$$v_{t+1}^{\text{FNO}} = \sigma\left(W_t(v_t)(x) + \mathcal{F}^{-1}\left(R_\phi \cdot \mathcal{F}(v_t)\right)(x) + b_t(x)\right)$$

Captures smooth, long-range, globally coherent features efficiently via $O(n \log n)$ FFT.

### 2. Wavelet (WNO) Branch — Local multi-resolution learning

Applies DWT (Daubechies wavelets), applies learnable weights in wavelet space, then applies inverse DWT:

$$v_{t+1}^{\text{WNO}} = \sigma\left(W_t(v_t)(x) + \mathcal{W}^{-1}\left(R_\phi \cdot \mathcal{W}(v_t)\right)(x) + b_t(x)\right)$$

Captures sharp gradients, local discontinuities, and multi-scale structures simultaneously localized in space and frequency.

### 3. Adaptive Gated Fusion Mechanism (AGFM)

The two branch outputs $v_{\text{FNO}}, v_{\text{WNO}} \in \mathbb{R}^{B \times H \times W \times C}$ are fused via a learned spatial gate:

$$\alpha = \sigma\left(\text{Conv}_{n \times n}([v_{\text{FNO}}, v_{\text{WNO}}])\right)$$

$$v_{\text{fused}} = \alpha \odot v_{\text{FNO}} + (1 - \alpha) \odot v_{\text{WNO}}$$

Where $\alpha \approx 1$ in smooth regions (FNO dominates) and $\alpha \approx 0$ near sharp features (WNO dominates). Empirically, a spatial gate ($C_{\text{gated}} = 1$) with a $1 \times 1$ convolutional filter works well.

---

<!-- ## Theoretical Foundation

Grounded in **Multiresolution Analysis (MRA)**, any $f \in L^2(\mathbb{R})$ can be decomposed as:

$$f(x) = \sum_{k} \langle f, \phi_{j_0,k}\rangle\, \phi_{j_0,k}(x) + \sum_{j \ge j_0} \sum_{k} \langle f, \psi_{j,k}\rangle\, \psi_{j,k}(x)$$

where $\phi$ (scaling function) captures smooth approximations and $\psi$ (mother wavelet) captures localized detail at each scale $j$. The FNO branch approximates the first sum; the WNO branch approximates the second.

--- -->

## Results

All models trained with Adam optimizer, 500 epochs, learning rate $10^{-3}$, StepLR scheduler (step 100, factor 0.5), weight decay $10^{-4}$, 4 layers per branch, on an NVIDIA RTX A6000 GPU.

| Benchmark | Metric | AW-FNO | FNO | WNO |
|-----------|--------|--------|-----|-----|
| Navier–Stokes 2D | Max pointwise abs. error | **0.0569** | 0.0589 | 1.4872 |
| Navier–Stokes 2D | Mean rel. $L_2$ error | 0.0195 | **0.0169** | 0.5180 |
| Burgers (discontinuity) | Mean rel. $L_2$ error | **0.0007** | 0.0315 | 0.0032 |
| Darcy (triangular notch) | Mean rel. $L_2$ error | **0.0047** | 0.0083 | 0.0057 |
| HIT super-resolution 4× | PSNR / SSIM | **57.40 dB / 0.9991** | 54.38 dB / 0.9985 | 48.98 dB / 0.9944 |

---

## Benchmarks

### 1D Burgers' Equation with Moving Discontinuity
Maps vorticity over the first 11 time steps to the next 40 steps. AW-FNO tracks the sharp jump near $x=0$ more closely than FNO (which smooths it due to mode truncation) while matching smooth regions as well as WNO.

### 2D Darcy Flow on a Triangular Notch Domain
Maps boundary conditions to pressure field on an irregular geometry with a slit interior boundary. AW-FNO attains lowest error, showing the gated fusion holds up with both smooth bulk regions and sharp notch features.

### 2D Navier–Stokes on a Periodic Square Domain
DNS data at $1024 \times 1024$ downsampled to $128 \times 128$, viscosity $\nu \in [2 \times 10^{-4}, 10^{-3}]$. AW-FNO achieves the lowest maximum pointwise error, reducing WNO's worst-case error by more than 25×.

### 4× Super-Resolution of Homogeneous Isotropic Turbulence (HIT)
Maps $32 \times 32$ low-resolution input to $128 \times 128$ high-resolution field. AW-FNO outperforms both FNO and WNO baselines, recovering fine-scale vortical structures with high fidelity.

---

## Repository Structure

```
AW-FNO/
├── awfno/                  # Core library
│   ├── models/             # AW-FNO, FNO, WNO model definitions
│   ├── layers/             # SpectralConv, WaveletConv, and supporting layers
│   └── utils/              # Losses, normalization, scaling utilities
├── examples/               # Per-benchmark training scripts (awfno/fno/wno × benchmark)
├── results/                # Training logs and model checkpoints
├── requirements.txt        # Python dependencies
├── setup.py                # Installation script
└── README.md
```

---

## Installation

```bash
git clone https://github.com/mamta-zenteiq-ai/AW-FNO.git
cd AW-FNO
pip install -r requirements.txt
pip install -e .
```

---

## Citation

If you use this work, please cite:

```
@inproceedings{saini2026awfno,
  title     = {AW-FNO: An Adaptive Approach for Fluid Flow Super-Resolution
               via Gated Wavelet-Fourier Learning},
  author    = {Saini, Mamta and Mahajan, Parikshit and Jyrwa, Gazania Marine
               and Nagchaudhury, Diya},
  booktitle = {Thirteenth International Conference on Computational Fluid
               Dynamics (ICCFD13)},
  address   = {Milan, Italy},
  year      = {2026},
  note      = {Paper ICCFD13-2026-311}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
