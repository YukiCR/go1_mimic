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
