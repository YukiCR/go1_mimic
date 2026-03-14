#!/usr/bin/env python3
"""Analyze HDF5 dataset structure and verify depth_image validity."""

import h5py
import numpy as np
import sys

def analyze_hdf5_structure(file_path):
    """Recursively analyze HDF5 file structure."""

    def print_structure(name, obj, indent=0):
        prefix = "  " * indent
        if isinstance(obj, h5py.Group):
            print(f"{prefix}[Group] {name}/")
            for key, val in obj.attrs.items():
                print(f"{prefix}  @{key}: {val}")
        elif isinstance(obj, h5py.Dataset):
            print(f"{prefix}[Dataset] {name}: shape={obj.shape}, dtype={obj.dtype}")
            for key, val in obj.attrs.items():
                print(f"{prefix}  @{key}: {val}")

    with h5py.File(file_path, 'r') as f:
        print(f"=== HDF5 File Structure: {file_path} ===\n")
        print(f"Root keys: {list(f.keys())}\n")

        # Recursively visit all items
        def visitor(name, obj):
            depth = name.count('/')
            print_structure(name, obj, depth)

        f.visititems(visitor)

        # Also check if there's a data group
        if 'data' in f:
            print("\n=== Data Group Details ===")
            data_group = f['data']
            print(f"Demos in data: {list(data_group.keys())}")

            # Check first demo structure
            demo_keys = list(data_group.keys())
            if demo_keys:
                first_demo = data_group[demo_keys[0]]
                print(f"\nFirst demo ({demo_keys[0]}) structure:")
                print(f"  Keys: {list(first_demo.keys())}")

                if 'obs' in first_demo:
                    obs_group = first_demo['obs']
                    print(f"  Obs keys: {list(obs_group.keys())}")

                    # Check depth_image structure
                    if 'depth_image' in obs_group:
                        depth_img = obs_group['depth_image']
                        print(f"\n  depth_image dataset:")
                        print(f"    Shape: {depth_img.shape}")
                        print(f"    Dtype: {depth_img.dtype}")

if __name__ == "__main__":
    dataset_path = "/home/chengrui/wk/ILBL_isaac/go1_mimic/go1_mimic/datasets/dataset.hdf5"
    analyze_hdf5_structure(dataset_path)
