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

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="DAgger data collection with human intervention.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to robomimic checkpoint.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dagger_demos.hdf5", help="File to save demonstrations."
)
parser.add_argument("--intervention_key", type=str, default="P", help="Key to trigger intervention.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to collect.")
parser.add_argument("--horizon", type=int, default=80, help="Step horizon for each rollout.")
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

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg


def create_env_with_recorder(task_name: str, dataset_path: str) -> tuple[gym.Env, object]:
    """Create environment configured for demonstration recording.

    Args:
        task_name: Name of the task/environment.
        dataset_path: Path to save recorded demonstrations.

    Returns:
        Tuple of (configured environment, success term).
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


def setup_teleop_device(intervention_callback: callable) -> Se2Keyboard:
    """Set up keyboard teleoperation device with intervention callback.

    Args:
        intervention_callback: Function to call when intervention key is pressed.

    Returns:
        Configured teleop device.
    """
    teleop_cfg = Se2KeyboardCfg(
        v_x_sensitivity=0.8,
        v_y_sensitivity=0.8,
    )
    teleop_interface = Se2Keyboard(teleop_cfg)

    # Register the intervention callback
    teleop_interface.add_callback(args_cli.intervention_key.upper(), intervention_callback)

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


def main():
    """Main entry point for DAgger data collection."""
    # State constants
    STATE_POLICY = "policy"
    STATE_HUMAN = "human"

    # Shared state variables (mutable containers for closure access)
    current_state = [STATE_POLICY]
    episode_count = [0]
    step_count = [0]
    intervention_requested = [False]

    def on_intervention():
        """Callback when intervention key is pressed."""
        if current_state[0] == STATE_POLICY:
            intervention_requested[0] = True
            print(f"[INFO] Intervention requested! Switching to HUMAN control.")

    # Set seeds
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    # Get device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print(f"[INFO] Using device: {device}")

    # Load policy
    print(f"[INFO] Loading policy from: {args_cli.checkpoint}")
    policy, _ = load_policy(args_cli.checkpoint, device)

    # Create environment with recorder
    print(f"[INFO] Creating environment: {args_cli.task}")
    env, success_term = create_env_with_recorder(args_cli.task, args_cli.dataset_file)
    env.seed(args_cli.seed)

    # Setup teleop device with intervention callback
    print(f"[INFO] Setting up teleop device (intervention key: '{args_cli.intervention_key}')")
    teleop_interface = setup_teleop_device(on_intervention)

    print(f"[INFO] Starting DAgger collection. Press '{args_cli.intervention_key}' to intervene.")
    print(f"[INFO] Target: {args_cli.num_episodes} human-controlled episodes")
    print("=" * 80)

    # Reset environment and policy
    obs_dict, _ = env.reset()
    policy.start_episode()
    teleop_interface.reset()

    # Main collection loop
    while episode_count[0] < args_cli.num_episodes and simulation_app.is_running():
        # Check for state transition (intervention requested)
        if intervention_requested[0]:
            intervention_requested[0] = False
            current_state[0] = STATE_HUMAN
            teleop_interface.reset() # reset the teleop interface to avoid stale inputs
            env.recorder_manager.reset() # reset recorder to start fresh for human episode, do not record policy steps
            print(f"[INFO] Now in HUMAN control mode (episode {episode_count[0] + 1})")

        # Get action based on current state
        if current_state[0] == STATE_POLICY:
            # Preprocess observations
            obs = preprocess_observations(obs_dict, env)

            # Get action from policy
            action = policy(obs)

            # Unnormalize if needed
            if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
                action = (
                    (action + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
                ) / 2 + args_cli.norm_factor_min

            action = torch.from_numpy(action).to(device=device).view(1, env.action_space.shape[1])

        else:  # STATE_HUMAN
            # Get action from teleop device
            # print(f"[DEBUG] Waiting for human action input...")
            action = teleop_interface.advance()
            action = action.repeat(env.num_envs, 1)

        # Step environment
        obs_dict, reward, terminated, truncated, info = env.step(action)
        step_count[0] += 1

        # Check success condition
        is_success = False
        if success_term is not None:
            is_success = bool(success_term.func(env, **success_term.params)[0])

        # Handle episode end
        if is_success or terminated or truncated or step_count[0] >= args_cli.horizon:
            if current_state[0] == STATE_HUMAN and is_success:
                # Human-controlled episode succeeded - record it
                print(f"[INFO] Human-controlled episode {episode_count[0] + 1} succeeded! Exporting...")

                # Mark as successful and export
                env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                env.recorder_manager.set_success_to_episodes(
                    [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                )
                env.recorder_manager.export_episodes([0])

                episode_count[0] += 1
                print(f"[INFO] Progress: {episode_count[0]}/{args_cli.num_episodes} episodes collected")

            elif current_state[0] == STATE_HUMAN:
                # Human-controlled episode failed - discard
                print(f"[INFO] Human-controlled episode failed. Discarding...")
                env.recorder_manager.reset()

            else:
                # Policy-controlled episode ended - just log
                status = "success" if is_success else "failure"
                print(f"[INFO] Policy-controlled episode ended ({status}). Resetting...")

            # Reset for next episode
            obs_dict, _ = env.reset()
            policy.start_episode()
            teleop_interface.reset()
            current_state[0] = STATE_POLICY
            step_count[0] = 0

    print("=" * 80)
    print(f"[INFO] Collection complete. {episode_count[0]} human demonstrations recorded.")

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
