import torch
from awfno.models.fno import FNO
from awfno.models.wno import WNO3d
from awfno.models.awfno_v2 import AWFNOv2_3d

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Navier-Stokes (3D) Parameters:")

# FNO 3D
fno = FNO(
    n_modes=(1, 12, 12),
    in_channels=1,
    out_channels=1,
    hidden_channels=128,
    n_layers=4,
    use_channel_mlp=False
)
print(f"FNO-3d:     {count_params(fno):,} parameters")

# WNO 3D
wno = WNO3d(
    in_channels=1,
    out_channels=1,
    width=16,
    size=(1, 64, 64),
    level=3,
    n_layers=4,
    wavelet='db6'
)
print(f"WNO-3d:     {count_params(wno):,} parameters")

# AW-FNO v2 3D
awfno = AWFNOv2_3d(
    in_channels=1,
    out_channels=1,
    n_modes=(1, 12, 12),
    size=(1, 64, 64),
    hidden_channels=16,
    n_fno_layers=4,
    n_wno_layers=4,
    wno_wavelet='db6'
)
print(f"AW-FNOv2-3d: {count_params(awfno):,} parameters")
