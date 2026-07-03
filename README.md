# AW-FNO: Adaptive Wavelet-Fourier Neural Operator

> **Paper:** *AW-FNO: An Adaptive approach for Fluid flow Super-resolution via Gated Wavelet-Fourier Learning*
> Mamta Saini, Parikshit Mahajan, Gazania Marine Jyrwa, Diya Nagchaudhury
> ICCFD13, Milan, Italy, July 2026

---

## Overview

**AW-FNO** fuses the **Fourier Neural Operator** (FNO) and the **Wavelet Neural Operator** (WNO)
with a learned **Adaptive Gated Fusion Mechanism** (GFM).

| Branch | Strength | Limitation (alone) |
|---|---|---|
| FNO (Fourier) | Global long-range correlations | Gibbs oscillations near sharp gradients |
| WNO (Wavelet) | Localised multi-scale features | Misses global coherent structures |
| **AW-FNO** | **Both — spatially decided** | — |

The gate α(x,y) ∈ (0,1) learns *where* to trust each branch:

```
V_fused = α ⊙ V_FNO  +  (1 − α) ⊙ V_WNO
α = σ(Conv1×1([V_FNO, V_WNO]))     ← per-spatial-location, per-channel
```

- **α ≈ 1** (smooth, large-scale flow) → FNO dominates
- **α ≈ 0** (vortex cores, shear layers) → WNO dominates

---

## Architecture

```
Input (B, C_in, H, W)
  │  GridEmbedding2D  →  append (x,y) coords
  │  Lifting MLP      →  (B, C_hid, H, W)
  │
  ├─ [AWFNOBlock2d] × L
  │    │ SpectralConv2d  →  V_fno    # FFT → filter → IFFT
  │    │ WaveConv2d      →  V_wno    # DWT → filter → IDWT
  │    │ GFM: α = σ(Conv1×1([V_fno, V_wno]))
  │    │ V_out = α⊙V_fno + (1−α)⊙V_wno + skip
  │    └ LayerNorm + GELU
  │
  └  Projection MLP   →  (B, C_out, H, W)
```

---

<!-- ## Results (Navier-Stokes 2D, Re=1000, 64×64)

| Model | Rel L2 ↓ | MSE ↓ | Params |
|---|---|---|---|
| FNO | — | — | — |
| WNO | — | — | — |
| AW-FNO (no gate) | — | — | — |
| **AW-FNO (ours)** | **—** | **—** | — |

*Run `make paper-ready` to populate this table with trained checkpoints.*

--- -->

## Quick Start

### 1. Install

```bash
git clone https://github.com/mamta-zenteiq-ai/AW-FNO.git
cd AW-FNO

# Conda (recommended)
conda env create -f environment.yml
conda activate aw-fno

# Or pip
pip install -r requirements.txt
```

### 2. Get data

```bash
# Download FNO benchmark datasets (~300 MB)
python datasets/download_fno_data.py --dataset ns2d

# Or if you already have the data:
export DATA_PATH=/path/to/navier_stokes
```

### 3. Train a single model

```bash
# AW-FNO (proposed)
make train-awfno DATA_PATH=/path/to/data

# Baseline: FNO
make train-fno DATA_PATH=/path/to/data

# Baseline: WNO
make train-wno DATA_PATH=/path/to/data
```

### 4. Full paper reproduction

```bash
# Trains all models, runs benchmark, generates all figures and tables
DATA_PATH=/path/to/data make paper-ready
```

---

## Repository Structure

```
AW-FNO/
├── awfno/                    # Core package
│   ├── models/
│   │   ├── awfno.py          # AW-FNO v1 (per-layer GFM) — paper primary
│   │   ├── awfno_v2.py       # AW-FNO v2 (branch-parallel)
│   │   ├── fno.py            # FNO baseline
│   │   └── wno.py            # WNO baseline
│   ├── layers/               # SpectralConv, WaveConv, embeddings
│   └── utils/                # Normalisation, losses, seed
│
├── configs/
│   ├── model/                # Per-model YAML configs
│   ├── dataset/              # Dataset YAML configs
│   └── experiment/           # Full experiment YAML configs
│
├── datasets/
│   ├── ns2d.py               # NavierStokes2DDataset (PyTorch Dataset)
│   ├── burgers1d.py          # Burgers1DDataset
│   └── download_fno_data.py  # Automated dataset download
│
├── experiments/
│   ├── train.py              # Unified training entry-point
│   ├── evaluate.py           # Evaluation + metric reporting
│   ├── benchmark.py          # Side-by-side model comparison
│   └── ablation.py           # Ablation study driver
│
├── trainers/
│   └── operator_trainer.py   # Training loop (AMP, grad-clip, CSV log)
│
├── utils/
│   ├── metrics.py            # Rel-L2, MSE, MAE, spectral L2, MetricTracker
│   ├── losses.py             # LpLoss, H1Loss, CombinedLoss
│   ├── visualization.py      # Field plots, gate maps, PSD, convergence curves
│   └── logging.py            # CSVLogger, get_logger
│
├── scripts/
│   ├── reproduce_all.sh      # Full paper reproduction pipeline
│   └── generate_paper_figures.py  # Figures + LaTeX tables
│
├── tests/                    # Pytest unit tests
├── docs/                     # Research overview, dataset survey
├── outputs/
│   ├── figures/              # Generated paper figures (PDF/PNG)
│   └── tables/               # LaTeX-ready .tex table files
│
├── Makefile                  # Shortcuts for all common tasks
├── environment.yml           # Conda environment
├── requirements.txt          # Pip requirements
└── paper.tex                 # Paper source
```

---

## Experiments

### Train all models + run benchmark

```bash
# Full run (uses DATA_PATH env var)
bash scripts/reproduce_all.sh

# Individual steps
python experiments/train.py --config configs/experiment/train_awfno_ns.yaml \
    --data_path /path/to/data --epochs 500

python experiments/benchmark.py --data_path /path/to/data --save_figures
```

### Ablation study

```bash
# Run no-gate variant
python experiments/train.py --config configs/experiment/ablation_no_gate.yaml \
    --data_path /path/to/data

# Collect all ablation results
python experiments/ablation.py --data_path /path/to/data
```

### Evaluate a checkpoint

```bash
python experiments/evaluate.py \
    --checkpoint results/awfno_ns/best.pt \
    --config configs/experiment/train_awfno_ns.yaml \
    --data_path /path/to/data \
    --save_figures
```

---

## Configuration

All experiments are controlled by YAML files in `configs/`. Override any field via CLI:

```bash
python experiments/train.py \
    --config configs/experiment/train_awfno_ns.yaml \
    --epochs 200 \
    --lr 5e-4 \
    --output_dir results/awfno_ns_lr5e4
```

Key experiment config fields:

| Field | Default | Description |
|---|---|---|
| `epochs` | 500 | Training epochs |
| `learning_rate` | 1e-3 | Adam LR |
| `scheduler_step_size` | 100 | StepLR decay step |
| `scheduler_gamma` | 0.5 | StepLR decay factor |
| `amp` | false | Automatic mixed precision |
| `grad_clip` | 1.0 | Gradient clipping max-norm |
| `seed` | 42 | Global random seed |

---

## Tests

```bash
make test                    # Run all unit tests
python -m pytest tests/ -v   # Verbose
```

Tests cover: model forward passes, gradient flow, loss functions, metrics, dataset loading,
normalizer roundtrips, and the no-gate ablation patch.

---

## Citation

```bibtex
@inproceedings{saini2026awfno,
  title     = {{AW-FNO}: An Adaptive approach for Fluid flow Super-resolution
               via Gated Wavelet-Fourier Learning},
  author    = {Saini, Mamta and Mahajan, Parikshit and Jyrwa, Gazania Marine
               and Nagchaudhury, Diya and Ganesan, Sashikumaar},
  booktitle = {Proceedings of the 13th International Conference on
               Computational Fluid Dynamics (ICCFD13)},
  year      = {2026},
  address   = {Milan, Italy},
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

This work uses:
- [neuraloperator](https://github.com/neuraloperator/neuraloperator) — spectral convolution layers
- [pytorch-wavelets](https://github.com/fbcotter/pytorch_wavelets) — DWT implementation
- [tensorly](https://github.com/tensorly/tensorly) / [tltorch](https://github.com/tensorly/torch) — tensor decomposition
- FNO dataset from [Li et al. (2021)](https://arxiv.org/abs/2010.08895)
