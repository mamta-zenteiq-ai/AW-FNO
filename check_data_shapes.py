import torch
import os

data_path = '/media/HDD/mamta_backup/datasets/fno/navier_stokes'
train_data = torch.load(os.path.join(data_path, 'ns_train_64.pt'))
test_data  = torch.load(os.path.join(data_path, 'ns_test_64.pt'))

print("Train X:", train_data['x'].shape)
print("Train Y:", train_data['y'].shape)
print("Test X:", test_data['x'].shape)
print("Test Y:", test_data['y'].shape)
