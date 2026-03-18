# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate DAgger algorithm across multiple checkpoints and datasets.

This script evaluates a sequence of DAgger checkpoints and computes metrics:
- Success rate vs DAgger roll
- Data amount vs DAgger roll

Input Modes:
    Mode 1 (default): Read checkpoint and dataset paths from a text file.
        Format: One checkpoint path per line, followed by a separator line,
                then one dataset path per line, in corresponding order.
    Mode 2: Directly specify checkpoints and datasets via command line arguments.

Args:
    input_file: Path to text file containing checkpoint and dataset paths (Mode 1).
    checkpoints: List of checkpoint paths (Mode 2).
    datasets: List of dataset paths corresponding to checkpoints (Mode 2).
    task: Name of the task/environment.
    num_rollouts: Number of rollouts per checkpoint for success rate evaluation.
    horizon: Step horizon for each rollout.
    output_dir: Directory to save evaluation outputs (report and plots).
    seed: Random seed.
    norm_factor_min: Minimum action normalization factor.
    norm_factor_max: Maximum action normalization factor.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate DAgger algorithm across multiple checkpoints.")

# Input mode 1: Text file
parser.add_argument(
    "--input_file",
    type=str,
    default=None,
    help="Path to text file containing checkpoint paths (first section) and dataset paths (second section). "
         "Sections are separated by a line containing only '---'.",
)

# Input mode 2: Direct paths
parser.add_argument(
    "--checkpoints",
    type=str,
    nargs="+",
    default=None,
    help="List of checkpoint paths for each DAgger iteration.",
)
parser.add_argument(
    "--datasets",
    type=str,
    nargs="+",
    default=None,
    help="List of dataset paths for each DAgger iteration (corresponding to checkpoints).",
)

# Common arguments
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_rollouts", type=int, default=10, help="Number of rollouts per checkpoint.")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon for each rollout.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="./dagger_eval_results",
    help="Directory to save evaluation results (report and plots).",
)
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum normalization factor."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Validate input arguments
if args_cli.input_file is not None:
    if args_cli.checkpoints is not None or args_cli.datasets is not None:
        parser.error("Cannot use --input_file together with --checkpoints or --datasets.")
else:
    if args_cli.checkpoints is None or args_cli.datasets is None:
        parser.error("Must provide either --input_file or both --checkpoints and --datasets.")
    if len(args_cli.checkpoints) != len(args_cli.datasets):
        parser.error("Number of checkpoints must match number of datasets.")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import gymnasium as gym
import numpy as np
import random
import torch
import h5py
from typing import List, Tuple

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

import go1_mimic.tasks
from isaaclab_tasks.utils import parse_env_cfg

import matplotlib.pyplot as plt
import gc


def parse_input_file(file_path: str) -> Tuple[List[str], List[str]]:
    """Parse input file containing checkpoint and dataset paths.

    Format:
        checkpoint_path_1
        checkpoint_path_2
        ...
        ---
        dataset_path_1
        dataset_path_2
        ...

    Lines starting with '#' are treated as comments and ignored.
    Empty lines are also ignored.

    Args:
        file_path: Path to the input file.

    Returns:
        Tuple of (list of checkpoint paths, list of dataset paths).
    """
    with open(file_path, 'r') as f:
        # Read lines, strip whitespace, skip empty lines and comment lines (starting with #)
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

    # Find separator
    separator_idx = None
    for i, line in enumerate(lines):
        if line == '---':
            separator_idx = i
            break

    if separator_idx is None:
        raise ValueError("Input file must contain a separator line '---' between checkpoints and datasets.")

    checkpoints = lines[:separator_idx]
    datasets = lines[separator_idx + 1:]

    if len(checkpoints) != len(datasets):
        raise ValueError(f"Number of checkpoints ({len(checkpoints)}) must match number of datasets ({len(datasets)}).")

    return checkpoints, datasets


def get_dataset_size(dataset_path: str) -> int:
    """Get the total number of samples in a dataset file.

    Args:
        dataset_path: Path to the HDF5 dataset file.

    Returns:
        Total number of samples (sum of demo lengths).
    """
    if not os.path.exists(dataset_path):
        print(f"[WARNING] Dataset file not found: {dataset_path}")
        return 0

    try:
        with h5py.File(dataset_path, 'r') as f:
            if 'data' not in f:
                print(f"[WARNING] No 'data' group found in {dataset_path}")
                return 0

            data_group = f['data']
            total_samples = 0

            for demo_key in data_group.keys():
                demo = data_group[demo_key]
                if 'actions' in demo:
                    total_samples += demo['actions'].shape[0]
                elif 'obs' in demo:
                    # Try to infer length from observations
                    obs_group = demo['obs']
                    if len(obs_group.keys()) > 0:
                        first_obs = list(obs_group.values())[0]
                        total_samples += first_obs.shape[0]

            return total_samples
    except Exception as e:
        print(f"[WARNING] Error reading dataset {dataset_path}: {e}")
        return 0


def create_env(task_name: str, device: str):
    """Create environment for evaluation.

    Args:
        task_name: Name of the task/environment.
        device: Device to run on.

    Returns:
        Tuple of (environment, success_term).
    """
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Set observations to dictionary mode for Robomimic
    env_cfg.observations.policy.concatenate_terms = False

    # Disable timeout termination
    env_cfg.terminations.time_out = None

    # Disable recorder
    env_cfg.recorders = None

    # Extract success checking function
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    # Create environment
    env = gym.make(task_name, cfg=env_cfg).unwrapped

    return env, success_term


def preprocess_observations(obs_dict: dict, env: gym.Env) -> dict:
    """Preprocess observations for robomimic policy inference.

    Args:
        obs_dict: Raw observation dictionary from environment.
        env: Environment instance.

    Returns:
        Preprocessed observation dictionary.
    """
    obs = copy.deepcopy(obs_dict["policy"])

    # Squeeze batch dimension
    for key in obs:
        obs[key] = torch.squeeze(obs[key])

    # Process image observations if present
    if hasattr(env.cfg, "image_obs_list"):
        for image_name in env.cfg.image_obs_list:
            if image_name in obs_dict["policy"].keys():
                image = torch.squeeze(obs_dict["policy"][image_name])
                image = image.permute(2, 0, 1).clone().float()
                image = image / 255.0
                image = image.clip(0.0, 1.0)
                obs[image_name] = image

    # Handle depth_image shape if present
    if "depth_image" in obs:
        if len(obs["depth_image"].shape) == 2:
            obs["depth_image"] = obs["depth_image"].unsqueeze(0)

    # Handle lader_distance shape if present
    if "lader_distance" in obs:
        if len(obs["lader_distance"].shape) == 1:
            obs["lader_distance"] = obs["lader_distance"].unsqueeze(0)

    return obs


def rollout_policy(policy, env, success_term, horizon: int, device) -> bool:
    """Perform a single rollout and check if successful.

    Args:
        policy: The robomimic policy.
        env: The environment.
        success_term: Success termination condition.
        horizon: Maximum rollout horizon.
        device: Device to run on.

    Returns:
        True if rollout was successful, False otherwise.
    """
    policy.start_episode()
    obs_dict, _ = env.reset()

    with torch.no_grad():
        for _ in range(horizon):
            # Preprocess observations
            obs = preprocess_observations(obs_dict, env)

            # Compute actions
            actions = policy(obs)

            # Unnormalize actions
            if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
                actions = (
                    (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
                ) / 2 + args_cli.norm_factor_min

            actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

            # Apply actions
            obs_dict, _, terminated, truncated, _ = env.step(actions)

            # Check if rollout was successful
            if bool(success_term.func(env, **success_term.params)[0]):
                return True
            elif terminated or truncated:
                return False

    return False


def evaluate_checkpoint(
    checkpoint_path: str,
    env: gym.Env,
    success_term,
    num_rollouts: int,
    horizon: int,
    device: torch.device,
) -> float:
    """Evaluate a single checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        env: The environment.
        success_term: Success termination condition.
        num_rollouts: Number of rollouts to perform.
        horizon: Maximum rollout horizon.
        device: Device to run on.

    Returns:
        Success rate (fraction of successful rollouts).
    """
    print(f"  Loading policy from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"  [WARNING] Checkpoint not found: {checkpoint_path}")
        return 0.0

    try:
        policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=checkpoint_path,
            device=device,
            verbose=False,
        )
    except Exception as e:
        print(f"  [WARNING] Error loading checkpoint: {e}")
        return 0.0

    results = []
    for trial in range(num_rollouts):
        with torch.no_grad():
            policy.start_episode()
            success = rollout_policy(policy, env, success_term, horizon, device)
        results.append(success)
        print(f"    Trial {trial}: {'SUCCESS' if success else 'FAILURE'}")

    success_rate = results.count(True) / len(results)
    print(f"  Success rate: {success_rate:.2%} ({results.count(True)}/{len(results)})")

    return success_rate


def plot_results(
    dagger_rolls: np.ndarray,
    success_rates: np.ndarray,
    data_amounts: np.ndarray,
    output_dir: str,
):
    """Generate evaluation plots.

    Args:
        dagger_rolls: Array of DAgger roll numbers (x-axis).
        success_rates: Array of success rates.
        data_amounts: Array of data amounts (number of samples).
        output_dir: Directory to save plots.
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Success Rate vs DAgger Roll
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(dagger_rolls, success_rates, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel("DAgger Roll", fontsize=12)
    ax1.set_ylabel("Success Rate", fontsize=12)
    ax1.set_title("Success Rate vs DAgger Roll", fontsize=14)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(dagger_rolls)

    plot1_path = os.path.join(output_dir, "success_rate_vs_roll.png")
    fig1.savefig(plot1_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved plot: {plot1_path}")
    plt.close(fig1)

    # Plot 2: Data Amount vs DAgger Roll
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(dagger_rolls, data_amounts, marker='s', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel("DAgger Roll", fontsize=12)
    ax2.set_ylabel("Data Amount (samples)", fontsize=12)
    ax2.set_title("Data Amount vs DAgger Roll", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(dagger_rolls)

    plot2_path = os.path.join(output_dir, "data_amount_vs_roll.png")
    fig2.savefig(plot2_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved plot: {plot2_path}")
    plt.close(fig2)


def generate_report(
    dagger_rolls: np.ndarray,
    success_rates: np.ndarray,
    data_amounts: np.ndarray,
    checkpoint_paths: List[str],
    dataset_paths: List[str],
    output_path: str,
):
    """Generate text report of evaluation results.

    Args:
        dagger_rolls: Array of DAgger roll numbers.
        success_rates: Array of success rates.
        data_amounts: Array of data amounts.
        checkpoint_paths: List of checkpoint paths.
        dataset_paths: List of dataset paths.
        output_path: Path to save the report.
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DAgger Evaluation Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Task: {args_cli.task}\n")
        f.write(f"Number of rollouts per checkpoint: {args_cli.num_rollouts}\n")
        f.write(f"Horizon per rollout: {args_cli.horizon}\n")
        f.write(f"Seed: {args_cli.seed}\n\n")

        f.write("-" * 80 + "\n")
        f.write("Results Summary\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"{'Roll':<8} {'Success Rate':<15} {'Data Amount':<15} {'Checkpoint':<30}\n")
        f.write("-" * 80 + "\n")

        for i, roll in enumerate(dagger_rolls):
            checkpoint_name = os.path.basename(checkpoint_paths[i])
            f.write(f"{roll:<8} {success_rates[i]:<15.4f} {data_amounts[i]:<15} {checkpoint_name:<30}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Detailed Paths\n")
        f.write("=" * 80 + "\n\n")

        for i, roll in enumerate(dagger_rolls):
            f.write(f"Roll {roll}:\n")
            f.write(f"  Checkpoint: {checkpoint_paths[i]}\n")
            f.write(f"  Dataset:    {dataset_paths[i]}\n")
            f.write(f"  Success Rate: {success_rates[i]:.4f}\n")
            f.write(f"  Data Amount:  {data_amounts[i]} samples\n\n")

    print(f"[INFO] Saved report: {output_path}")


def main():
    """Main entry point for DAgger evaluation."""
    # Parse input
    if args_cli.input_file is not None:
        print(f"[INFO] Reading input from file: {args_cli.input_file}")
        checkpoint_paths, dataset_paths = parse_input_file(args_cli.input_file)
    else:
        checkpoint_paths = args_cli.checkpoints
        dataset_paths = args_cli.datasets

    num_rolls = len(checkpoint_paths)
    print(f"[INFO] Evaluating {num_rolls} DAgger iterations")

    # Create output directory
    os.makedirs(args_cli.output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {args_cli.output_dir}")

    # Set seeds
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    # Get device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print(f"[INFO] Using device: {device}")

    # Create environment (reused for all checkpoints)
    print(f"[INFO] Creating environment: {args_cli.task}")
    env, success_term = create_env(args_cli.task, args_cli.device)
    env.seed(args_cli.seed)

    # Set seeds again after environment creation
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    # Evaluate each checkpoint
    dagger_rolls = np.arange(1, num_rolls + 1)
    success_rates = np.zeros(num_rolls)
    data_amounts = np.zeros(num_rolls, dtype=int)

    print("\n" + "=" * 80)
    print("Starting Evaluation")
    print("=" * 80 + "\n")

    for i in range(num_rolls):
        print(f"[{i + 1}/{num_rolls}] Evaluating DAgger Roll {i + 1}")
        print(f"  Checkpoint: {checkpoint_paths[i]}")
        print(f"  Dataset: {dataset_paths[i]}")

        # Evaluate checkpoint
        success_rates[i] = evaluate_checkpoint(
            checkpoint_paths[i],
            env,
            success_term,
            args_cli.num_rollouts,
            args_cli.horizon,
            device,
        )

        # Get dataset size
        data_amounts[i] = get_dataset_size(dataset_paths[i])
        print(f"  Dataset size: {data_amounts[i]} samples")
        print()

        # Memory cleanup between checkpoints
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    env.close()

    # Generate outputs
    print("=" * 80)
    print("Generating Outputs")
    print("=" * 80 + "\n")

    # Plot results
    plot_results(dagger_rolls, success_rates, data_amounts, args_cli.output_dir)

    # Generate report
    report_path = os.path.join(args_cli.output_dir, "dagger_evaluation_report.txt")
    generate_report(
        dagger_rolls,
        success_rates,
        data_amounts,
        checkpoint_paths,
        dataset_paths,
        report_path,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"{'Roll':<8} {'Success Rate':<15} {'Data Amount':<15}")
    print("-" * 40)
    for i in range(num_rolls):
        print(f"{dagger_rolls[i]:<8} {success_rates[i]:<15.4f} {data_amounts[i]:<15}")

    print(f"\nResults saved to: {args_cli.output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
