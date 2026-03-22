#!/usr/bin/env python3
"""Filter out episodes with all-zero depth images and create a clean dataset."""

import h5py
import numpy as np
import os

def copy_group_recursive(src_group, dst_group):
    """Recursively copy a group from source to destination."""

    # Copy attributes
    for key, val in src_group.attrs.items():
        dst_group.attrs[key] = val

    for key in src_group.keys():
        src_item = src_group[key]

        if isinstance(src_item, h5py.Group):
            # Create new group and recursively copy
            new_group = dst_group.create_group(key)
            copy_group_recursive(src_item, new_group)
        elif isinstance(src_item, h5py.Dataset):
            # Copy dataset data and attributes
            dst_group.create_dataset(key, data=src_item[:], dtype=src_item.dtype,
                                     compression=src_item.compression,
                                     compression_opts=src_item.compression_opts)
            # Copy dataset attributes
            for attr_key, attr_val in src_item.attrs.items():
                dst_group[key].attrs[attr_key] = attr_val

def filter_dataset(input_path, output_path, invalid_episodes):
    """Create a new dataset with invalid episodes removed."""

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Episodes to filter out: {len(invalid_episodes)}")
    print("-" * 60)

    with h5py.File(input_path, 'r') as src_f:
        with h5py.File(output_path, 'w') as dst_f:

            # Copy root attributes
            for key, val in src_f.attrs.items():
                dst_f.attrs[key] = val

            # Process data group
            src_data = src_f['data']
            dst_data = dst_f.create_group('data')

            # Copy data group attributes (except total)
            for key, val in src_data.attrs.items():
                if key != 'total':
                    dst_data.attrs[key] = val

            # Copy all valid demos
            demo_keys = sorted(src_data.keys(), key=lambda x: int(x.split('_')[1]))
            kept_count = 0

            for demo_key in demo_keys:
                if demo_key in invalid_episodes:
                    print(f"  Filtering out: {demo_key}")
                    continue

                # Copy this demo
                src_demo = src_data[demo_key]
                dst_demo = dst_data.create_group(demo_key)

                # Copy attributes
                for attr_key, attr_val in src_demo.attrs.items():
                    dst_demo.attrs[attr_key] = attr_val

                # Copy datasets and subgroups recursively
                copy_group_recursive(src_demo, dst_demo)
                kept_count += 1

            # Update total count
            dst_data.attrs['total'] = kept_count

    print("-" * 60)
    print(f"Done! Kept {kept_count} episodes, removed {len(invalid_episodes)} episodes.")
    return kept_count

def verify_output(output_path, expected_count):
    """Verify the output dataset."""

    print("\nVerifying output dataset...")
    with h5py.File(output_path, 'r') as f:
        data_group = f['data']
        demos = list(data_group.keys())
        total = data_group.attrs.get('total', len(demos))

        print(f"  Total episodes in output: {total}")
        print(f"  Expected: {expected_count}")

        # Verify no all-zero depth images
        print("  Checking depth images are valid...")
        for demo_key in demos:
            depth_img = data_group[demo_key]['obs']['depth_image'][:]
            if np.all(depth_img == 0):
                print(f"    ERROR: {demo_key} still has all-zero depth_image!")
                return False

        print("  All depth images are valid!")
        return True

if __name__ == "__main__":
    input_path = "/home/chengrui/wk/ILBL_isaac/go1_mimic/go1_mimic/datasets/dataset.hdf5"
    output_path = "/home/chengrui/wk/ILBL_isaac/go1_mimic/go1_mimic/tempoutputs/dataset_filtered.hdf5"

    # Read invalid episodes list
    invalid_episodes_file = "/home/chengrui/wk/ILBL_isaac/go1_mimic/go1_mimic/tempoutputs/invalid_episodes.txt"
    with open(invalid_episodes_file, 'r') as f:
        invalid_episodes = [line.strip() for line in f if line.strip()]

    # Filter dataset
    kept = filter_dataset(input_path, output_path, invalid_episodes)

    # Verify output
    if verify_output(output_path, kept):
        print(f"\nFiltered dataset saved to: {output_path}")
    else:
        print("\nVerification failed!")
