import torch
import os

data_path = '/media/HDD/mamta_backup/datasets/fno/navier_stokes'
train_data = torch.load(os.path.join(data_path, 'nsforcing_train_128.pt'))
test_data  = torch.load(os.path.join(data_path, 'nsforcing_test_128.pt'))

if isinstance(train_data, dict):
    for k, v in train_data.items():
        if isinstance(v, torch.Tensor):
            print(f"Train {k}: {v.shape}")
else:
    print("Train shape:", train_data.shape)

if isinstance(test_data, dict):
    for k, v in test_data.items():
        if isinstance(v, torch.Tensor):
            print(f"Test {k}: {v.shape}")
else:
    print("Test shape:", test_data.shape)
