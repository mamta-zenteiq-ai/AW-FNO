"""
WNO baseline for SOD shock super-resolution.

Uses the WNO1d class from awfno/models/wno.py with hyperparameters matched
to the AW-FNO v2 configuration (same width=64, wavelet='db6', level=3).

Architecture:
  Input (B, 3, 256) → append grid → (B, 4, 256) → Linear lifting (B, 64, 256)
  → 4 × WaveletConv + Skip + GELU → Linear projection → (B, 3, 256)
  → ×4 linear interpolation → (B, 3, 1024)
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from awfno.models.wno import WNO1d
from sod_common import (
    DATA_ROOT, EPOCHS, BATCH_SIZE, LEARNING_RATE, LR_RESOLUTION,
    run_experiment,
)

MODEL_NAME  = 'wno_baseline'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'baselines', MODEL_NAME)

# ─── Hyper-parameters matching AW-FNO v2 WNO branch config ───────────────────
WIDTH   = 64
LEVEL   = 3
N_LAYERS = 4          # total WNO layers — matches (2 FNO + 2 WNO) combined
WAVELET = 'db6'


def build_model():
    return WNO1d(
        in_channels=3,
        out_channels=3,
        width=WIDTH,
        size=[LR_RESOLUTION],   # input spatial size (256)
        level=LEVEL,
        n_layers=N_LAYERS,
        padding=0,
        wavelet=WAVELET,
    )


if __name__ == '__main__':
    model = build_model()
    run_experiment(
        model_name=MODEL_NAME,
        base_model=model,
        results_dir=RESULTS_DIR,
        data_root=DATA_ROOT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        extra_meta={
            'architecture': 'WNO',
            'width': WIDTH,
            'level': LEVEL,
            'n_layers': N_LAYERS,
            'wavelet': WAVELET,
        },
    )
