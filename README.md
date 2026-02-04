# AW-FNO: Adaptive Wavelet-Fourier Neural Operator

Breaking the frequency-localization trade-off in SciML by unifying global Spectral methods with local Multiresolution Analysis via learnable spatial gating.

## Overview

**AW-FNO** is a next-generation Neural Operator designed to overcome the fundamental limitations of the vanilla Fourier Neural Operator (FNO). By decomposing the latent function space into global spectral approximations and local multiresolution details, AW-FNO accurately captures both smooth macro-physics and localized singularities (like shocks or phase transitions) without the artifacts of global spectral bias.

---

## Core Methodology: The Dual-Stream Framework

The AW-FNO architecture employs a dual-stream approach to represent complex physical fields:

### 1. The Fourier Stream (Global Low-Rank Approximation)
Acts as a low-pass filter to capture the "macro-physics." By truncating high-frequency modes ($k_{max}$), it maintains a low-rank representation that ensures global structural consistency and computational efficiency.
$$(\mathcal{K}v)(x) = \mathcal{F}^{-1}(R_{\phi} \cdot \mathcal{F}v)(x)$$

### 2. The Wavelet Stream (Local Singularity Tracking)
Utilizes Discrete/Stationary Wavelet Transforms (DWT/SWT). Unlike Fourier modes, wavelets are compactly supported in both space and frequency, allowing the model to resolve spatially varying frequencies and non-periodic boundary conditions without spectral leakage.

### 3. The Latent Spatial Gate (Adaptive Delegation)
A learnable spatial map $\alpha \in [0, 1]^{H \times W}$ computed from the local hidden state:
$$\alpha = \sigma(W_{gate} * h + b_{gate})$$
The final operator output is the gated sum:
$$\mathcal{K}_{AW}(h) = \alpha \odot \text{SpectralConv}(h) + (1 - \alpha) \odot \text{WaveletConv}(h)$$

---

## Repository Structure

```text
AW-FNO/
├── awfno/                  # Core library
│   ├── models/             # AW-FNO Model definitions
│   ├── layers/             # SpectralConv, WaveletConv, and Gating layers
│   └── utils/              # Spectral/Wavelet transforms and data processing
├── examples/               # Example scripts for various PDEs
│   ├── burgers2d/          # 2D Burgers' equation
│   ├── ginzburg_landau/    # Ginzburg-Landau equation (complex-valued)
│   └── schrodinger/        # Schrödinger equation
├── configs/                # Hyperparameter and model configurations (YAML/JSON)
├── data/                   # Dataset storage (gitignored)
├── docs/                   # Extended documentation and analysis
├── tests/                  # Unit tests for layers and models
├── scripts/                # Training, evaluation, and visualization scripts
├── README.md               # Project overview
├── requirements.txt        # Python dependencies
└── setup.py                # Installation script
```

---

## Theoretical Foundation

Grounded in **Multiresolution Analysis (MRA)**, AW-FNO approximates any function $f \in L^2(\mathbb{R})$ via its decomposition:
$$ f(x) = \underbrace{\sum_{k} c_k \phi_k(x)}_{\text{Fourier Stream}} + \underbrace{\sum_{j \ge 0} \sum_{k} d_{j,k} \psi_{j,k}(x)}_{\text{Wavelet Stream}} $$
This provides a more complete representation of the underlying Sobolev space than FNO alone.

---

## Evaluation Plan

We evaluate AW-FNO on complex fields where phase-localization is paramount:
- **Complex-Valued Ginzburg-Landau Equation**
- **Schrödinger Equation**

### Baselines
- Vanilla FNO
- WNO (Multiwavelets)
- U-Net based PDE solvers

### Metrics
- **Relative $L_2$ Error**: Global accuracy.
- **Maximum Error near Singularities**: Measuring the reduction in Gibbs oscillations.

---

## Installation

```bash
git clone https://github.com/mamta-zenteiq-ai/AW-FNO.git
cd AW-FNO
pip install -r requirements.txt
```

---

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
