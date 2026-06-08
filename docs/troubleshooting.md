# Troubleshooting

## Installation Issues

### `pytorch_wavelets` import error
```
ModuleNotFoundError: No module named 'pytorch_wavelets'
```
Fix:
```bash
pip install pytorch-wavelets
```
Note: the package name on PyPI uses a hyphen but is imported with an underscore.

### `tensorly` / `tltorch` version conflict
The spectral convolution layer requires compatible `tensorly` and `tltorch` versions:
```bash
pip install tensorly==0.8.1 tltorch==0.3.0
```

### CUDA out of memory during training
- Reduce `batch_size` in `configs/dataset/ns2d.yaml`
- Reduce `hidden_channels` in the model config
- Enable AMP: set `amp: true` in the experiment config
- Use gradient checkpointing (future feature)

---

## Data Issues

### Dataset not found
```
FileNotFoundError: No NS dataset files found in data/ns2d
```
Fix:
```bash
python datasets/download_fno_data.py --dataset ns2d
```
Or set `DATA_PATH` to an existing directory:
```bash
export DATA_PATH=/media/HDD/mamta_backup/datasets/fno/navier_stokes
```

### Wrong tensor shapes
The NS dataset files may be in different formats depending on source.
Run a quick check:
```python
import torch
d = torch.load('path/to/ns_train_64.pt')
print({k: v.shape for k, v in d.items()})
```
Expected: `{'x': torch.Size([1000, 1, 64, 64]), 'y': torch.Size([1000, 1, 64, 64])}`

---

## Training Issues

### Loss is NaN from epoch 1
- Check data normalisation — extreme values can cause NaN in wavelet transforms
- Reduce learning rate (try `lr: 1e-4`)
- Enable gradient clipping: `grad_clip: 0.5`

### WNO loss not decreasing
- The wavelet convolution weights are initialised with `scale=0.05` — they need
  a few epochs to warm up. Check that loss decreases after epoch 10.
- Ensure the wavelet branch is not dominated by the FNO branch (check gate maps).

### Gate maps are all 0.5 after training
- This can happen with very small models or very short training.
- The gate learns slowly — check at epoch 200+.
- Try a higher `hidden_channels` value (32 vs 8).

---

## Evaluation Issues

### `eval_metrics.json` missing
Run the evaluate script explicitly:
```bash
python experiments/evaluate.py \
    --checkpoint results/awfno_ns/best.pt \
    --config configs/experiment/train_awfno_ns.yaml \
    --data_path /path/to/data
```

### Gate map figure crashes
```
RuntimeError: No gate maps captured. Is this an AWFNO2d model?
```
Gate visualisation only works with `AWFNO2d` (v1). `AWFNOv2_2d` uses a
different hook location — update `plot_gate_maps` to hook into `model.gate`.

---

## CI / Test Issues

### Tests fail with `No module named 'awfno'`
Ensure the project root is on `PYTHONPATH`:
```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
python -m pytest tests/
```
Or install in editable mode:
```bash
pip install -e .
```

### `test_fixed_gate_ablation` fails with import error
```
ImportError: cannot import name '_patch_fixed_gate' from 'experiments.train'
```
The test imports directly from `experiments/train.py`. Ensure it is importable:
```bash
cd /path/to/AW-FNO && python -c "from experiments.train import _patch_fixed_gate"
```
