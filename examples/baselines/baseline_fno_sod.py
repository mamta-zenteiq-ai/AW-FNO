"""
FNO baseline for SOD shock super-resolution.

Uses the FNO class from neuralop (https://github.com/neuraloperator/neuraloperator)
with standard hyperparameters matched to the AW-FNO v2 hidden dimension (64)
and comparable depth (4 layers).

All training, evaluation, and plotting are handled by sod_common.run_experiment()
to guarantee identical conditions across all baselines.

Architecture:
  Input (B, 3, 256) → GridEmbedding (B, 4, 256) → Lifting (B, 64, 256)
  → 4 × FNOBlock (spectral conv + skip + GELU) → Projection → (B, 3, 256)
  → ×4 linear interpolation → (B, 3, 1024)
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch.nn.functional as F

from neuralop.models import FNO
from sod_common import (
    DATA_ROOT, EPOCHS, BATCH_SIZE, LEARNING_RATE, LR_RESOLUTION,
    run_experiment,
)

MODEL_NAME  = 'fno_baseline'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'baselines', MODEL_NAME)

# ─── Hyper-parameters matching AW-FNO v2 capacity ────────────────────────────
N_MODES         = (64,)    # Fourier modes — captures full spectrum for L=256
HIDDEN_CHANNELS = 64
N_LAYERS        = 4        # total FNO blocks — matches 2 FNO + 2 WNO in AWFNO v2


def build_model():
    return FNO(
        n_modes=N_MODES,
        in_channels=3,
        out_channels=3,
        hidden_channels=HIDDEN_CHANNELS,
        n_layers=N_LAYERS,
        positional_embedding='grid',     # appends normalised x coordinate
        non_linearity=F.gelu,
        use_channel_mlp=False,           # keep architecture close to original FNO
        fno_skip='linear',
        norm=None,
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
            'architecture': 'FNO',
            'n_modes': list(N_MODES),
            'hidden_channels': HIDDEN_CHANNELS,
            'n_layers': N_LAYERS,
        },
    )
