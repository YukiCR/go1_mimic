# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script for pure DAgger data collection with implicit intervention detection.

This script implements a DAgger-style data collection where intervention is detected
by comparing human input against zero action. When human provides non-zero input
for consecutive steps, it switches to human mode. When human stops providing input,
it exports the segment and returns to policy mode.

This generates (s_policy, a_human) pairs which are the core of DAgger algorithm.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    dataset_file: Path to export recorded demonstrations.
    num_segments: Number of intervention segments to collect.
    horizon: Maximum step horizon for each rollout.
    debounce_steps: Number of consecutive non-zero human actions to trigger intervention.
    min_segment_length: Minimum steps for a valid segment.
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
parser = argparse.ArgumentParser(description="Pure DAgger data collection with implicit intervention.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to robomimic checkpoint.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/pure_dagger_demos.hdf5", help="File to save demonstrations."
)
parser.add_argument("--num_segments", type=int, default=100, help="Number of intervention segments to collect.")
parser.add_argument("--horizon", type=int, default=50, help="Step horizon for each rollout.")
parser.add_argument(
    "--debounce_steps", type=int, default=2, help="Consecutive non-zero actions to trigger intervention."
)
parser.add_argument(
    "--min_segment_length", type=int, default=5, help="Minimum steps for a valid segment."
)
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum normalization factor."
)
parser.add_argument(
    "--reset_key", type=str, default="R", help="Key to press for resetting current episode."
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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
import isaaclab.utils.math as math_utils


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


def setup_teleop_device(reset_callback: callable) -> Se2Keyboard:
    """Set up keyboard teleoperation device.

    Args:
        reset_callback: Function to call when reset key is pressed.

    Returns:
        Configured teleop device.
    """
    teleop_cfg = Se2KeyboardCfg(
        v_x_sensitivity=0.8,
        v_y_sensitivity=0.8,
    )
    teleop_interface = Se2Keyboard(teleop_cfg)

    # Register the reset callback
    teleop_interface.add_callback(args_cli.reset_key.upper(), reset_callback)

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

    # TODO: The following observation preprocessing is environment-specific.
    # These shape fixes assume specific observation structures (depth_image, lader_distance).
    # For general use, this should be made configurable or use the environment's observation spec.
    # Handle depth_image shape if present
    if "depth_image" in obs:
        if len(obs["depth_image"].shape) == 2:
            obs["depth_image"] = obs["depth_image"].unsqueeze(0)

    # Handle lader_distance shape if present
    if "lader_distance" in obs:
        if len(obs["lader_distance"].shape) == 1:
            obs["lader_distance"] = obs["lader_distance"].unsqueeze(0)

    return obs


def is_zero_action(action: torch.Tensor, tolerance: float = 0.01) -> bool:
    """Check if action is effectively zero.

    Args:
        action: Action tensor.
        tolerance: Tolerance for zero check.

    Returns:
        True if action is close to zero.
    """
    return torch.all(torch.abs(action) < tolerance).item()


def export_segment(env: gym.Env, segment_step_count: int, total_segments: int) -> int:
    """Export the current recorded segment.

    Args:
        env: Environment instance.
        segment_step_count: Number of steps in the segment.
        total_segments: Current count of exported segments.

    Returns:
        Updated total segments count.
    """
    print(f"[INFO] Exporting segment with {segment_step_count} steps...")

    # Mark as successful and export
    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    env.recorder_manager.set_success_to_episodes(
        [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
    )
    env.recorder_manager.export_episodes([0])

    total_segments += 1
    print(f"[INFO] Progress: {total_segments} segments collected")
    return total_segments


def init_human_action_visualizer(env: gym.Env) -> VisualizationMarkers:
    """Initialize red arrow visualizer for human actions.

    Args:
        env: Environment instance.

    Returns:
        Configured visualization markers for human actions.
    """
    # Red arrow for human action
    human_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/HumanAction/velocity_human")
    human_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    human_visualizer = VisualizationMarkers(human_cfg)
    return human_visualizer


def visualize_human_action(env: gym.Env, human_visualizer: VisualizationMarkers, a_human: torch.Tensor) -> None:
    """Visualize human action as a red arrow.

    Args:
        env: Environment instance.
        human_visualizer: Human action visualizer.
        a_human: Human action tensor [1, 3] with [vx, vy, vyaw].
    """
    if human_visualizer is None:
        return

    # Get robot base position
    base_pos = env.scene["robot"].data.root_pos_w.clone()  # [1, 3]
    base_pos[:, 2] += 0.6  # Offset up for visibility (above robot)

    # Resolve velocity command to arrow
    xy_velocity = a_human[:, :2].to(env.device)  # [1, 2]

    # Obtain default scale of the marker
    default_scale = human_visualizer.cfg.markers["arrow"].scale

    # Arrow scale based on velocity magnitude
    arrow_scale = torch.tensor(default_scale, device=env.device).repeat(xy_velocity.shape[0], 1)
    vel_magnitude = torch.linalg.norm(xy_velocity, dim=1)
    arrow_scale[:, 0] *= vel_magnitude * 5.0

    # Arrow direction (yaw from velocity)
    heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])  # [1,]
    zeros = torch.zeros_like(heading_angle)
    arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)  # [1, 4]

    # Convert from base to world frame
    base_quat_w = env.scene["robot"].data.root_quat_w  # [1, 4]
    arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

    # Visualize
    human_visualizer.visualize(base_pos, arrow_quat, arrow_scale)


def main():
    """Main entry point for pure DAgger data collection."""
    # State constants
    STATE_POLICY = "policy"
    STATE_HUMAN = "human"

    # Initialize state variables
    current_state = STATE_POLICY
    debounce_counter = 0
    segment_step_count = 0
    total_segments = 0
    total_step_count = 0
    reset_requested = False

    def on_reset():
        """Callback when reset key is pressed."""
        nonlocal reset_requested
        reset_requested = True
        print(f"[INFO] Reset requested by user (key: '{args_cli.reset_key}').")

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

    # Setup teleop device with reset callback
    print(f"[INFO] Setting up teleop device (reset key: '{args_cli.reset_key}')")
    teleop_interface = setup_teleop_device(on_reset)

    # Initialize human action visualizer
    print(f"[INFO] Initializing human action visualizer")
    human_visualizer = init_human_action_visualizer(env)

    print(f"[INFO] Starting Pure DAgger collection.")
    print(f"[INFO] Target: {args_cli.num_segments} segments")
    print(f"[INFO] Debounce: {args_cli.debounce_steps} steps, Min segment: {args_cli.min_segment_length} steps")
    print(f"[INFO] Press '{args_cli.reset_key}' to reset current episode")
    print("=" * 80)

    # Reset environment and policy
    obs_dict, _ = env.reset()
    policy.start_episode()
    teleop_interface.reset()

    # Main collection loop
    while total_segments < args_cli.num_segments and simulation_app.is_running():
        # Always get both actions
        # Policy action (for inference and RNN state maintenance)
        obs = preprocess_observations(obs_dict, env)
        a_policy_raw = policy(obs)

        # Unnormalize if needed
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            a_policy_raw = (
                (a_policy_raw + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        a_policy = torch.from_numpy(a_policy_raw).to(device=device).view(1, env.action_space.shape[1])

        # Human action
        a_human_raw = teleop_interface.advance()
        a_human = a_human_raw.repeat(env.num_envs, 1)

        # State machine for action selection
        if current_state == STATE_POLICY:
            # Check for human input (non-zero action)
            if not is_zero_action(a_human_raw):
                debounce_counter += 1
                if debounce_counter >= args_cli.debounce_steps:
                    # Switch to HUMAN mode
                    current_state = STATE_HUMAN
                    segment_step_count = 0
                    env.recorder_manager.reset()  # Start fresh segment
                    action = a_human
                    print(f"[INFO] Switched to HUMAN mode (segment {total_segments + 1})")
                else:
                    # Still debouncing, use policy action
                    action = a_policy
            else:
                # Reset debounce counter
                debounce_counter = 0
                action = a_policy
        else:  # STATE_HUMAN
            # Check if human stopped providing input
            if is_zero_action(a_human_raw):
                # Export segment if long enough
                if segment_step_count >= args_cli.min_segment_length:
                    total_segments = export_segment(env, segment_step_count, total_segments)
                else:
                    print(f"[INFO] Discarding short segment ({segment_step_count} < {args_cli.min_segment_length})")
                    env.recorder_manager.reset()

                # Switch back to POLICY mode
                current_state = STATE_POLICY
                debounce_counter = 0
                action = a_policy  # Use policy action, not zero
                print(f"[INFO] Switched to POLICY mode")
            else:
                # Continue in human mode
                action = a_human
                segment_step_count += 1

        # Step environment
        obs_dict, reward, terminated, truncated, info = env.step(action)
        total_step_count += 1

        # Visualize human action (always show to reflect when human is intervening)
        visualize_human_action(env, human_visualizer, a_human)

        # Check success condition
        is_success = False
        if success_term is not None:
            is_success = bool(success_term.func(env, **success_term.params)[0])

        # Handle episode end (including user reset request)
        episode_ended = is_success or terminated or truncated or total_step_count >= args_cli.horizon or reset_requested

        if episode_ended:
            if current_state == STATE_HUMAN:
                # Export segment if in human mode
                if segment_step_count >= args_cli.min_segment_length:
                    total_segments = export_segment(env, segment_step_count, total_segments)
                else:
                    print(f"[INFO] Discarding short segment at episode end ({segment_step_count} < {args_cli.min_segment_length})")
                    env.recorder_manager.reset()

            # Log episode status
            if reset_requested:
                print(f"[INFO] Episode reset by user")
            elif is_success:
                print(f"[INFO] Episode ended with success")
            elif terminated or truncated:
                print(f"[INFO] Episode ended (terminated/truncated)")
            else:
                print(f"[INFO] Episode ended (horizon reached)")

            # Reset for new rollout
            obs_dict, _ = env.reset()
            policy.start_episode()
            current_state = STATE_POLICY
            debounce_counter = 0
            segment_step_count = 0
            total_step_count = 0
            reset_requested = False

    print("=" * 80)
    print(f"[INFO] Collection complete. {total_segments} segments recorded.")

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
