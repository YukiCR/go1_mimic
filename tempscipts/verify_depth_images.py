#!/usr/bin/env python3
"""Verify depth_image validity - check for zero matrices in each episode."""

import h5py
import numpy as np
import sys

def check_depth_images(file_path):
    """Check all episodes for zero depth_images."""

    invalid_episodes = []
    total_episodes = 0
    zero_stats = []

    with h5py.File(file_path, 'r') as f:
        data_group = f['data']
        demo_keys = sorted(data_group.keys(), key=lambda x: int(x.split('_')[1]))
        total_episodes = len(demo_keys)

        print(f"Total episodes to check: {total_episodes}")
        print("Checking depth_image validity...")
        print("-" * 60)

        for i, demo_key in enumerate(demo_keys):
            demo_group = data_group[demo_key]
            obs_group = demo_group['obs']

            if 'depth_image' not in obs_group:
                print(f"[{i+1}/{total_episodes}] {demo_key}: WARNING - no depth_image found")
                invalid_episodes.append(demo_key)
                continue

            depth_image = obs_group['depth_image'][:]

            # Check if all frames are zero matrices
            is_all_zero = np.all(depth_image == 0)
            has_zero_frames = np.any(np.all(depth_image == 0, axis=(1, 2, 3)))

            # Statistics
            min_val = np.min(depth_image)
            max_val = np.max(depth_image)
            mean_val = np.mean(depth_image)
            zero_count = np.sum(depth_image == 0)
            total_pixels = depth_image.size

            zero_stats.append({
                'demo': demo_key,
                'is_all_zero': is_all_zero,
                'has_zero_frames': has_zero_frames,
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'zero_ratio': zero_count / total_pixels
            })

            if is_all_zero:
                print(f"[{i+1}/{total_episodes}] {demo_key}: INVALID - all zero matrix!")
                invalid_episodes.append(demo_key)
            elif has_zero_frames:
                zero_frames = np.sum(np.all(depth_image == 0, axis=(1, 2, 3)))
                print(f"[{i+1}/{total_episodes}] {demo_key}: PARTIAL - {zero_frames}/{depth_image.shape[0]} frames are zero")
            else:
                if (i+1) % 1000 == 0 or i == 0:
                    print(f"[{i+1}/{total_episodes}] {demo_key}: OK (min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f})")

    print("-" * 60)
    print(f"\nSUMMARY:")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Invalid episodes (all zero): {len(invalid_episodes)}")

    if invalid_episodes:
        print(f"\n  Invalid episode list:")
        for ep in invalid_episodes:
            print(f"    - {ep}")

    # Additional statistics
    all_zero_count = sum(1 for s in zero_stats if s['is_all_zero'])
    partial_zero_count = sum(1 for s in zero_stats if s['has_zero_frames'] and not s['is_all_zero'])

    print(f"\n  Episodes with all zero frames: {all_zero_count}")
    print(f"  Episodes with some zero frames: {partial_zero_count}")

    return invalid_episodes, zero_stats

if __name__ == "__main__":
    dataset_path = "/home/chengrui/wk/ILBL_isaac/go1_mimic/go1_mimic/datasets/dataset.hdf5"
    invalid_episodes, stats = check_depth_images(dataset_path)

    # Save invalid episodes list
    if invalid_episodes:
        output_file = "/home/chengrui/wk/ILBL_isaac/go1_mimic/go1_mimic/tempoutputs/invalid_episodes.txt"
        with open(output_file, 'w') as f:
            for ep in invalid_episodes:
                f.write(f"{ep}\n")
        print(f"\nInvalid episodes list saved to: {output_file}")
