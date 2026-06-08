# Benchmarking Guide

## Standard Protocol

All models must be compared under identical conditions:
- Same dataset, same train/test split (seed 42)
- Same normalisation (UnitGaussian on train set, applied to test)
- Same number of epochs (500)
- Same optimiser: Adam, lr=1e-3, weight_decay=1e-4
- Same scheduler: StepLR, step=100, γ=0.5
- Same seed for weight initialisation (seed 42)
- Same batch size (20)
- Same loss function (relative L2)

## Running the Benchmark

```bash
# 1. Train all models (sequential, ~2–3 hrs on a single A100)
make train-all DATA_PATH=/path/to/ns/data

# 2. Run ablation
make ablation DATA_PATH=/path/to/ns/data

# 3. Collect results into a table
python experiments/benchmark.py --data_path /path/to/data --save_figures

# 4. Generate LaTeX
python scripts/generate_paper_figures.py --data_path /path/to/data
```

## Metrics Definition

### Primary: Relative L2 Error

$$\text{Rel}\ L_2 = \frac{1}{N} \sum_{i=1}^{N} \frac{\|\hat{u}_i - u_i\|_2}{\|u_i\|_2}$$

Computed on the **physical** (unnormalised) fields after decoding.

### Secondary

| Metric | Formula | Notes |
|---|---|---|
| MSE | mean((pred−gt)²) | Absolute scale |
| MAE | mean(|pred−gt|) | Robust |
| Spectral L2 | ‖F(pred)−F(gt)‖ / ‖F(gt)‖ | Frequency-resolved |
| Max pointwise err | max|pred−gt| | Gibbs spikes |

## Expected Training Times

| Model | GPU | Approx time (500 ep) |
|---|---|---|
| FNO (C=32) | A100 40GB | ~2.5 hrs |
| WNO (w=32) | A100 40GB | ~3.5 hrs |
| AW-FNO (C=32, v1) | A100 40GB | ~3.5 hrs |
| AW-FNO (C=32, v2) | A100 40GB | ~4.0 hrs |

## Checkpoint Format

All checkpoints are `.pt` files saved by `OperatorTrainer` with:

```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "metrics": {"rel_l2": float, "mse": float, ...},
    "experiment_name": str,
}
```

## Fairness Checklist

Before reporting results:
- [ ] All models use the same `size` parameter (same spatial grid)
- [ ] `x_normalizer` and `y_normalizer` fitted only on train set
- [ ] `y_normalizer.decode()` applied before computing all metrics
- [ ] Seeds match across all runs
- [ ] Best checkpoint (not final) used for evaluation
- [ ] `model.eval()` and `torch.no_grad()` during evaluation

## Known Pitfalls

1. **WNO grid size**: WaveConv layers require spatial size to be divisible by 2^level.
   At level=2 and size=64: 64 / 4 = 16 ✓. At level=3: 64 / 8 = 8 ✓.
   Padding the input is required for non-power-of-2 sizes.

2. **FNO n_modes**: Must be ≤ resolution // 2. For 64×64, max n_modes = 32.
   Paper uses 12 (conservative; higher modes may overfit).

3. **Normalisation leak**: Never refit normalizer on test data. Always pass
   `x_normalizer` from train dataset to test dataset constructor.

4. **Decode before metrics**: The model outputs normalised predictions.
   `y_normalizer.decode(pred)` must be called before computing any metric.
