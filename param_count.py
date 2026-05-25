import torch
from awfno.models.fno import FNO
from awfno.models.wno import WNO1d, WNO2d
from awfno.models.awfno import AWFNO1d, AWFNO2d

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

fno_1d = FNO(n_modes=(16,), in_channels=1, out_channels=1, hidden_channels=64, n_layers=4, use_channel_mlp=True)
wno_1d = WNO1d(in_channels=1, out_channels=1, width=64, size=(128,), level=3, n_layers=4)
awfno_1d = AWFNO1d(in_channels=1, out_channels=1, n_modes=(16,), size=(128,), hidden_channels=64, n_layers=4)

print("1D Burgers Models:")
print(f"FNO:    {count_parameters(fno_1d):,} parameters")
print(f"WNO:    {count_parameters(wno_1d):,} parameters")
print(f"AW-FNO: {count_parameters(awfno_1d):,} parameters")

fno_2d = FNO(n_modes=(12,12), in_channels=1, out_channels=1, hidden_channels=64, n_layers=4, use_channel_mlp=True)
wno_2d = WNO2d(in_channels=1, out_channels=1, width=64, size=(64,64), level=3, n_layers=4)
awfno_2d = AWFNO2d(in_channels=1, out_channels=1, n_modes=(12,12), size=(64,64), hidden_channels=64, n_layers=4)

print("\n2D Navier-Stokes Models:")
print(f"FNO:    {count_parameters(fno_2d):,} parameters")
print(f"WNO:    {count_parameters(wno_2d):,} parameters")
print(f"AW-FNO: {count_parameters(awfno_2d):,} parameters")
