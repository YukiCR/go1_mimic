# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script for DAgger data collection with human intervention.

This script loads a pre-trained robomimic policy and allows human intervention
during rollout. When the human intervenes, they take full control until episode
success, and the demonstration is recorded to an HDF5 file.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    dataset_file: Path to export recorded demonstrations.
    intervention_key: Keyboard key to trigger human intervention.
    num_episodes: Number of human-intervened episodes to collect.
    horizon: Maximum step horizon for each rollout.
    seed: Random seed.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import gymnasium as gym
import numpy as np
import random
import torch
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="DAgger data collection with human intervention.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to robomimic checkpoint.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dagger_demos.hdf5", help="File to save demonstrations."
)
parser.add_argument("--intervention_key", type=str, default="space", help="Key to trigger intervention.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to collect.")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon for each rollout.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum normalization factor."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

import go1_mimic.tasks
from isaaclab_tasks.utils import parse_env_cfg

print("[INFO] DAgger collector imports successful")


def create_env_with_recorder(task_name: str, dataset_path: str) -> gym.Env:
    """Create environment configured for demonstration recording.

    Args:
        task_name: Name of the task/environment.
        dataset_path: Path to save recorded demonstrations.

    Returns:
        Configured environment with recorder manager.
    """
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode
    import os

    # Parse environment config
    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=1, use_fabric=True)

    # Set observations to dictionary mode for robomimic
    env_cfg.observations.policy.concatenate_terms = False

    # Disable timeout termination (we want episodes to run until success/intervention)
    env_cfg.terminations.time_out = None

    # Extract success term for checking
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    # Configure recorder manager
    output_dir = os.path.dirname(dataset_path)
    output_file_name = os.path.splitext(os.path.basename(dataset_path))[0]

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.export_in_record_pre_reset = False  # We'll handle export manually

    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create environment
    env = gym.make(task_name, cfg=env_cfg).unwrapped

    return env, success_term


def setup_teleop_device(intervention_key: str) -> object:
    """Set up keyboard teleoperation device.

    Args:
        intervention_key: Key that triggers intervention.

    Returns:
        Configured teleop device.
    """
    from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

    teleop_cfg = Se2KeyboardCfg(
        v_x_sensitivity=0.8,
        v_y_sensitivity=0.8,
    )
    teleop_interface = Se2Keyboard(teleop_cfg)

    return teleop_interface


def load_policy(checkpoint_path: str, device: torch.device):
    """Load robomimic policy from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load policy on.

    Returns:
        Loaded policy and checkpoint dictionary.
    """
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=checkpoint_path,
        device=device,
        verbose=False,
    )
    return policy, ckpt_dict


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
