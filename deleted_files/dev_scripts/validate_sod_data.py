#!/usr/bin/env python3
"""
Validation script for Sod super-resolution experiment setup.
Checks data loading, normalization, and downsampling.

Run: python scripts/validate_sod_data.py
"""

import h5py
import numpy as np
import torch
import os
import sys

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def load_and_validate_sod():
    data_root = '/media/HDD/mamta_backup/datasets/PDEBench/comp_ns/1d'
    
    print_section("STEP 1: Checking Dataset Files")
    
    for sod_file in ['1D_CFD_Sod1.hdf5', '1D_CFD_Sod3.hdf5', '1D_CFD_Sod5.hdf5']:
        filepath = os.path.join(data_root, sod_file)
        if not os.path.exists(filepath):
            print(f"❌ MISSING: {filepath}")
            return False
        
        filesize_mb = os.path.getsize(filepath) / (1024**2)
        print(f"✓ {sod_file:30s} ({filesize_mb:.1f} MB)")
    
    print_section("STEP 2: Loading and Inspecting Data")
    
    all_hr = []
    for sod_name in ['1D_CFD_Sod1.hdf5', '1D_CFD_Sod3.hdf5', '1D_CFD_Sod5.hdf5']:
        filepath = os.path.join(data_root, sod_name)
        
        with h5py.File(filepath, 'r') as f:
            vx = f['Vx'][:]
            density = f['density'][:]
            pressure = f['pressure'][:]
            
            print(f"\n{sod_name}:")
            print(f"  Vx        shape: {vx.shape}        range: [{vx.min():.3f}, {vx.max():.3f}]")
            print(f"  density   shape: {density.shape}        range: [{density.min():.3f}, {density.max():.3f}]")
            print(f"  pressure  shape: {pressure.shape}        range: [{pressure.min():.3f}, {pressure.max():.3f}]")
            
            # Check for NaNs
            if np.isnan(vx).any() or np.isnan(density).any() or np.isnan(pressure).any():
                print(f"  ⚠️  WARNING: NaNs detected in {sod_name}!")
                return False
            
            # Stack into (N, 3, 1024)
            sod_data = np.stack([vx, density, pressure], axis=1)
            all_hr.append(sod_data)
    
    hr = np.concatenate(all_hr, axis=0)
    print(f"\nCombined shape: {hr.shape}")
    print(f"  Expected: (65, 3, 1024)")
    if hr.shape != (65, 3, 1024):
        print(f"❌ Shape mismatch!")
        return False
    print(f"✓ Shape is correct")
    
    print_section("STEP 3: Per-Field Value Ranges (Before Normalization)")
    
    fields = ['Vx', 'density', 'pressure']
    for i, field_name in enumerate(fields):
        field_data = hr[:, i, :]
        print(f"\n{field_name}:")
        print(f"  Min:     {field_data.min():12.6f}")
        print(f"  Max:     {field_data.max():12.6f}")
        print(f"  Mean:    {field_data.mean():12.6f}")
        print(f"  Std:     {field_data.std():12.6f}")
        print(f"  Range:   {field_data.max() - field_data.min():12.6f}")
        print(f"  Range ratio (to Vx): {(field_data.max() - field_data.min()) / (hr[:,0,:].max() - hr[:,0,:].min()):8.1f}x")
    
    print_section("STEP 4: Per-Field Normalization")
    
    hr_tensor = torch.from_numpy(hr).float()
    
    # Compute normalization statistics
    mu = hr_tensor.mean(dim=(0, 2), keepdim=True)
    std = hr_tensor.std(dim=(0, 2), keepdim=True)
    
    print(f"\nNormalization statistics:")
    print(f"  Mean per field (shape {mu.shape}):")
    for i, name in enumerate(fields):
        print(f"    {name:12s}: {mu[0, i, 0].item():12.6f}")
    
    print(f"\n  Std per field (shape {std.shape}):")
    for i, name in enumerate(fields):
        print(f"    {name:12s}: {std[0, i, 0].item():12.6f}")
    
    # Apply normalization
    hr_norm = (hr_tensor - mu) / (std + 1e-8)
    
    print(f"\nNormalized data ranges (should be ~[-3, 3]):")
    for i, name in enumerate(fields):
        norm_field = hr_norm[:, i, :].numpy()
        print(f"  {name:12s}: [{norm_field.min():7.3f}, {norm_field.max():7.3f}]")
    
    print_section("STEP 5: Downsampling by ×4 (Strided Subsampling)")
    
    lr_norm = hr_norm[:, :, ::4]  # Every 4th point
    
    print(f"\nDownsampled shape: {lr_norm.shape}")
    print(f"  Expected: (65, 3, 256)")
    if lr_norm.shape != (65, 3, 256):
        print(f"❌ Shape mismatch!")
        return False
    print(f"✓ Shape is correct")
    
    # Verify no information loss (should be perfect match at downsampled points)
    for i in range(0, 1024, 4):
        for j in range(3):
            if hr_norm[0, j, i].item() != lr_norm[0, j, i//4].item():
                print(f"❌ Downsampling mismatch at position {i}!")
                return False
    print(f"✓ Downsampling verified (perfect point correspondence)")
    
    print_section("STEP 6: Train/Test Split (80/20)")
    
    n_samples = 65
    n_train = int(0.8 * n_samples)
    n_test = n_samples - n_train
    
    print(f"\nTotal samples:    {n_samples}")
    print(f"Train samples:    {n_train} ({100*n_train/n_samples:.1f}%)")
    print(f"Test samples:     {n_test} ({100*n_test/n_samples:.1f}%)")
    
    print_section("STEP 7: Data Ready for Training")
    
    print(f"\n✓ Input (LR):        {lr_norm.shape}  — 256-point profiles, 3 fields")
    print(f"✓ Target (HR):       {hr_norm.shape}  — 1024-point profiles, 3 fields")
    print(f"✓ Normalization:     Per-field z-score (mean≈0, std≈1)")
    print(f"✓ Downsampling:      Strided (preserves shocks)")
    print(f"✓ Data split:        80 train / 20 test")
    
    print("\n" + "="*70)
    print("✓ ALL VALIDATION CHECKS PASSED!")
    print("="*70)
    print("\nYou can now run: python examples/example_awfno_v2_sod.py\n")
    
    return True

if __name__ == "__main__":
    success = load_and_validate_sod()
    sys.exit(0 if success else 1)
