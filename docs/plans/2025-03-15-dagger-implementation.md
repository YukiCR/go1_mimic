# DAgger Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the DAgger algorithm for imitation learning with interactive human intervention and policy fine-tuning.

**Architecture:** Two main components: (1) `dagger_collect.py` - rolls out a pre-trained policy with human intervention capability, records human demonstrations to HDF5; (2) `dagger_finetune.py` - loads a checkpoint and fine-tunes on extended dataset (original + new demos).

**Tech Stack:** IsaacLab (environment, recorder manager), robomimic (policy loading, training), PyTorch, Gymnasium

---

## Prerequisites

- IsaacLab installed at `/home/chengrui/IsaacLab`
- robomimic installed in conda env `env_isaaclab`
- go1_mimic environment configured
- Existing checkpoint and dataset available for testing

---

## Task 1: Create DAgger Collector Script - Boilerplate and Imports

**Files:**
- Create: `dagger/scripts/dagger_collect.py`

**Step 1: Create file with header and imports**

Create `dagger/scripts/dagger_collect.py` with the following structure:

```python
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
```

**Step 2: Verify imports work**

Run: `cd /home/chengrui/wk/ILBL_isaac/go1_mimic/go1_mimic && python dagger/scripts/dagger_collect.py --help`

Expected: Help message displayed, no import errors

**Step 3: Commit**

```bash
git add dagger/scripts/dagger_collect.py
git commit -m "feat(dagger): add dagger collector boilerplate and imports"
```

---

## Task 2: Implement Environment Setup with Recorder

**Files:**
- Modify: `dagger/scripts/dagger_collect.py`

**Step 1: Add environment creation function after imports**

Add this function to `dagger/scripts/dagger_collect.py`:

```python
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
```

**Step 2: Add teleop device setup function**

Add this function after `create_env_with_recorder`:

```python
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
```

**Step 3: Commit**

```bash
git add dagger/scripts/dagger_collect.py
git commit -m "feat(dagger): add environment and teleop setup functions"
```

---

## Task 3: Implement Policy Loading and Inference

**Files:**
- Modify: `dagger/scripts/dagger_collect.py`

**Step 1: Add policy loading function**

Add after the teleop setup function:

```python
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
```

**Step 2: Add observation preprocessing function**

Add after `load_policy`:

```python
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
```

**Step 3: Commit**

```bash
git add dagger/scripts/dagger_collect.py
git commit -m "feat(dagger): add policy loading and observation preprocessing"
```

---

## Task 4: Implement Main DAgger Collection Loop

**Files:**
- Modify: `dagger/scripts/dagger_collect.py`

**Step 1: Add the main collection function**

Add this function after the preprocessing function:

```python
def collect_dagger_demos(
    env: gym.Env,
    policy,
    teleop_interface,
    success_term,
    device: torch.device,
    num_episodes: int,
    horizon: int,
    intervention_key: str,
):
    """Main DAgger collection loop with human intervention.

    Args:
        env: Environment instance.
        policy: Loaded robomimic policy.
        teleop_interface: Teleoperation device.
        success_term: Success termination term.
        device: Device for tensor operations.
        num_episodes: Number of human-intervened episodes to collect.
        horizon: Maximum steps per rollout.
        intervention_key: Key to trigger intervention.
    """
    from pynput import keyboard

    # State constants
    STATE_POLICY = "policy"
    STATE_HUMAN = "human"

    # Initialize state
    current_state = STATE_POLICY
    episode_count = 0
    step_count = 0

    # Intervention flag (shared between keyboard listener and main loop)
    intervention_requested = [False]

    def on_key_press(key):
        """Callback for keyboard listener."""
        try:
            if key.char == intervention_key:
                intervention_requested[0] = True
                print(f"[INFO] Intervention requested!")
        except AttributeError:
            # Special keys don't have char attribute
            if intervention_key == "space" and key == keyboard.Key.space:
                intervention_requested[0] = True
                print(f"[INFO] Intervention requested!")

    # Start keyboard listener in background
    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()

    print(f"[INFO] Starting DAgger collection. Press '{intervention_key}' to intervene.")
    print(f"[INFO] Target: {num_episodes} human-controlled episodes")
    print("=" * 80)

    # Reset environment and policy
    obs_dict, _ = env.reset()
    policy.start_episode()
    teleop_interface.reset()

    try:
        while episode_count < num_episodes:
            # Get action based on current state
            if current_state == STATE_POLICY:
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

                # Check for intervention request
                if intervention_requested[0]:
                    intervention_requested[0] = False
                    current_state = STATE_HUMAN
                    print(f"[INFO] Switching to HUMAN control (episode {episode_count + 1})")
                    continue

            else:  # STATE_HUMAN
                # Get action from teleop device
                action = teleop_interface.advance()
                action = action.repeat(env.num_envs, 1)

            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            # Check success condition
            is_success = False
            if success_term is not None:
                is_success = bool(success_term.func(env, **success_term.params)[0])

            # Handle episode end
            if is_success or terminated or truncated or step_count >= horizon:
                if current_state == STATE_HUMAN and is_success:
                    # Human-controlled episode succeeded - record it
                    print(f"[INFO] Human-controlled episode {episode_count + 1} succeeded! Exporting...")

                    # Mark as successful and export
                    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                    env.recorder_manager.set_success_to_episodes(
                        [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes([0])

                    episode_count += 1
                    print(f"[INFO] Progress: {episode_count}/{num_episodes} episodes collected")

                elif current_state == STATE_HUMAN:
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
                current_state = STATE_POLICY
                step_count = 0

    finally:
        listener.stop()
        print("=" * 80)
        print(f"[INFO] Collection complete. {episode_count} human demonstrations recorded.")
```

**Step 2: Add main function**

Add at the end of the file:

```python
def main():
    """Main entry point for DAgger data collection."""
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

    # Setup teleop device
    print(f"[INFO] Setting up teleop device (intervention key: '{args_cli.intervention_key}')")
    teleop_interface = setup_teleop_device(args_cli.intervention_key)

    # Run collection loop
    collect_dagger_demos(
        env=env,
        policy=policy,
        teleop_interface=teleop_interface,
        success_term=success_term,
        device=device,
        num_episodes=args_cli.num_episodes,
        horizon=args_cli.horizon,
        intervention_key=args_cli.intervention_key,
    )

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

**Step 3: Commit**

```bash
git add dagger/scripts/dagger_collect.py
git commit -m "feat(dagger): implement main DAgger collection loop with intervention"
```

---

## Task 5: Create DAgger Fine-tuning Script

**Files:**
- Create: `dagger/scripts/dagger_finetune.py`

**Step 1: Create file with imports and argument parsing**

Create `dagger/scripts/dagger_finetune.py`:

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script for fine-tuning a robomimic policy with extended dataset.

This script loads a pre-trained checkpoint and fine-tunes it on a combined
dataset (original + new DAgger demonstrations).

Args:
    checkpoint: Path to pre-trained checkpoint.
    original_dataset: Path to original training dataset.
    new_dataset: Path to new DAgger demonstrations.
    output_dir: Directory to save fine-tuned checkpoints.
    epochs: Number of fine-tuning epochs.
    batch_size: Batch size for training.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app (required for imports)
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import argparse
import h5py
import json
import numpy as np
import os
import shutil
import sys
import torch
import traceback
from collections import OrderedDict
from torch.utils.data import DataLoader

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.algo import algo_factory
from robomimic.config import config_factory
from robomimic.utils.log_utils import DataLogger, PrintLogger

import go1_mimic.tasks


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune robomimic policy with extended dataset.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pre-trained checkpoint.")
    parser.add_argument("--original_dataset", type=str, required=True, help="Path to original dataset.")
    parser.add_argument("--new_dataset", type=str, required=True, help="Path to new DAgger dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (None = use config).")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate (None = use config).")
    parser.add_argument("--seed", type=int, default=101, help="Random seed.")
    return parser.parse_args()


print("[INFO] DAgger fine-tuning imports successful")
```

**Step 2: Add dataset merging function**

Add after imports:

```python
def merge_datasets(original_path: str, new_path: str, output_path: str) -> str:
    """Merge original and new datasets into a single HDF5 file.

    Args:
        original_path: Path to original dataset.
        new_path: Path to new DAgger demonstrations.
        output_path: Path for merged output file.

    Returns:
        Path to merged dataset file.
    """
    print(f"[INFO] Merging datasets...")
    print(f"  Original: {original_path}")
    print(f"  New: {new_path}")
    print(f"  Output: {output_path}")

    # Copy original as base
    shutil.copyfile(original_path, output_path)

    with h5py.File(output_path, "r+") as f_out:
        with h5py.File(new_path, "r") as f_new:
            # Get existing demo count
            existing_demos = [k for k in f_out["data"].keys() if k.startswith("demo_")]
            demo_count = len(existing_demos)

            # Get new demos
            new_demos = [k for k in f_new["data"].keys() if k.startswith("demo_")]

            print(f"[INFO] Original demos: {demo_count}, New demos: {len(new_demos)}")

            # Copy new demos with updated names
            for i, demo_name in enumerate(new_demos):
                new_demo_name = f"demo_{demo_count + i}"
                f_out.copy(f_new[f"data/{demo_name}"], f"data/{new_demo_name}")

            # Update data attributes
            total_demos = demo_count + len(new_demos)
            f_out["data"].attrs["total"] = total_demos

    print(f"[INFO] Merged dataset created with {total_demos} total demonstrations")
    return output_path
```

**Step 3: Add fine-tuning function**

Add after `merge_datasets`:

```python
def finetune_policy(
    checkpoint_path: str,
    dataset_path: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int = None,
    learning_rate: float = None,
    seed: int = 101,
):
    """Fine-tune policy on extended dataset.

    Args:
        checkpoint_path: Path to pre-trained checkpoint.
        dataset_path: Path to merged dataset.
        output_dir: Directory to save checkpoints.
        num_epochs: Number of fine-tuning epochs.
        batch_size: Optional batch size override.
        learning_rate: Optional learning rate override.
        seed: Random seed.
    """
    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt_dict = FileUtils.load_dict_from_checkpoint(checkpoint_path)

    # Restore config
    config, _ = FileUtils.config_from_checkpoint(algo_name=ckpt_dict["algo_name"], ckpt_dict=ckpt_dict)

    # Update config for fine-tuning
    config.train.data = dataset_path
    config.train.num_epochs = num_epochs
    config.train.seed = seed

    if batch_size is not None:
        config.train.batch_size = batch_size

    if learning_rate is not None:
        # Update learning rate in all optimizer configs
        for key in config.algo:
            if hasattr(config.algo[key], "lr"):
                config.algo[key].lr = learning_rate

    # Setup output directories
    config.train.output_dir = os.path.abspath(output_dir)
    log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)

    print(f"[INFO] Output directories:")
    print(f"  Logs: {log_dir}")
    print(f"  Checkpoints: {ckpt_dir}")

    # Get device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print(f"[INFO] Using device: {device}")

    # Initialize obs utils
    ObsUtils.initialize_obs_utils_with_config(config)

    # Get metadata from checkpoint
    shape_meta = ckpt_dict["shape_metadata"]
    env_meta = ckpt_dict["env_metadata"]

    # Create model
    print(f"[INFO] Creating {config.algo_name} model...")
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    # Load pre-trained weights
    print(f"[INFO] Loading pre-trained weights...")
    model.deserialize(ckpt_dict["model"])

    # Save config
    with open(os.path.join(log_dir, "..", "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Load training data
    print(f"[INFO] Loading training data from: {dataset_path}")
    trainset, validset = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()

    print(f"[INFO] Training set: {len(trainset)} samples")
    if validset is not None:
        print(f"[INFO] Validation set: {len(validset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
    )

    valid_loader = None
    if config.experiment.validate and validset is not None:
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=min(config.train.num_data_workers, 1),
            drop_last=True,
        )

    # Initialize logger
    data_logger = DataLogger(log_dir, config=config, log_tb=config.experiment.logging.log_tb)

    # Fine-tuning loop
    print("=" * 80)
    print(f"[INFO] Starting fine-tuning for {num_epochs} epochs...")
    print("=" * 80)

    best_valid_loss = None

    for epoch in range(1, num_epochs + 1):
        # Training epoch
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=config.experiment.epoch_every_n_steps,
        )
        model.on_epoch_end(epoch)

        # Log training metrics
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {step_log.get('Loss', 'N/A')}")
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record(f"Timing_Stats/Train_{k[5:]}", v, epoch)
            else:
                data_logger.record(f"Train/{k}", v, epoch)

        # Validation epoch
        if valid_loader is not None:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model,
                    data_loader=valid_loader,
                    epoch=epoch,
                    validate=True,
                    num_steps=config.experiment.validation_epoch_every_n_steps,
                )

            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/Valid_{k[5:]}", v, epoch)
                else:
                    data_logger.record(f"Valid/{k}", v, epoch)

            # Save best validation checkpoint
            if "Loss" in step_log:
                if best_valid_loss is None or step_log["Loss"] <= best_valid_loss:
                    best_valid_loss = step_log["Loss"]
                    ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch}_best_validation.pth")
                    TrainUtils.save_model(
                        model=model,
                        config=config,
                        env_meta=env_meta,
                        shape_meta=shape_meta,
                        ckpt_path=ckpt_path,
                    )

        # Periodic checkpoint saving
        if config.experiment.save.enabled:
            save_ckpt = False
            if config.experiment.save.every_n_epochs and epoch % config.experiment.save.every_n_epochs == 0:
                save_ckpt = True
            if epoch == num_epochs:  # Always save last epoch
                save_ckpt = True

            if save_ckpt:
                ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pth")
                TrainUtils.save_model(
                    model=model,
                    config=config,
                    env_meta=env_meta,
                    shape_meta=shape_meta,
                    ckpt_path=ckpt_path,
                )
                print(f"[INFO] Saved checkpoint: {ckpt_path}")

    # Close logger
    data_logger.close()
    print("=" * 80)
    print(f"[INFO] Fine-tuning complete!")
    print(f"[INFO] Checkpoints saved to: {ckpt_dir}")
```

**Step 4: Add main function**

Add at the end:

```python
def main():
    """Main entry point for fine-tuning."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create merged dataset
    merged_dataset_path = os.path.join(args.output_dir, "merged_dataset.hdf5")
    merge_datasets(args.original_dataset, args.new_dataset, merged_dataset_path)

    # Run fine-tuning
    try:
        finetune_policy(
            checkpoint_path=args.checkpoint,
            dataset_path=merged_dataset_path,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    except Exception as e:
        print(f"[ERROR] Fine-tuning failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    simulation_app.close()
```

**Step 5: Commit**

```bash
git add dagger/scripts/dagger_finetune.py
git commit -m "feat(dagger): add fine-tuning script for DAgger"
```

---

## Task 6: Test DAgger Collector

**Files:**
- Test: `dagger/scripts/dagger_collect.py`

**Step 1: Test import and help**

Run: `cd /home/chengrui/wk/ILBL_isaac/go1_mimic/go1_mimic && python dagger/scripts/dagger_collect.py --help`

Expected: Help message with all arguments displayed

**Step 2: Test with dry-run (if checkpoint available)**

If you have a checkpoint available, test basic loading:

```bash
python dagger/scripts/dagger_collect.py \
    --task ILBL-Go1-Mimic-Box-v0 \
    --checkpoint logs/robomimic/.../model_epoch_60.pth \
    --dataset_file datasets/test_dagger.hdf5 \
    --num_episodes 1 \
    --horizon 100 \
    --headless
```

Expected: Script runs, environment creates, policy loads

**Step 3: Commit**

```bash
git commit -m "test(dagger): verify collector script loads correctly"
```

---

## Task 7: Test Fine-tuning Script

**Files:**
- Test: `dagger/scripts/dagger_finetune.py`

**Step 1: Test import and help**

Run: `cd /home/chengrui/wk/ILBL_isaac/go1_mimic/go1_mimic && python dagger/scripts/dagger_finetune.py --help`

Expected: Help message with all arguments displayed

**Step 2: Commit**

```bash
git commit -m "test(dagger): verify fine-tune script loads correctly"
```

---

## Task 8: Create README Documentation

**Files:**
- Create: `dagger/README.md`

**Step 1: Create README with usage instructions**

Create `dagger/README.md`:

```markdown
# DAgger Implementation for go1_mimic

Dataset Aggregation (DAgger) algorithm implementation for imitation learning in IsaacLab.

## Overview

DAgger addresses the distribution mismatch between expert demonstrations and policy rollouts by:
1. Rolling out the current policy
2. Having a human intervene and take control when needed
3. Recording the human demonstrations
4. Adding these to the training dataset
5. Fine-tuning the policy on the extended dataset

## Scripts

### 1. dagger_collect.py

Collects human demonstrations by rolling out a pre-trained policy and allowing intervention.

**Usage:**
```bash
python dagger/scripts/dagger_collect.py \
    --task ILBL-Go1-Mimic-Box-v0 \
    --checkpoint logs/robomimic/.../model_epoch_60.pth \
    --dataset_file datasets/dagger_demos.hdf5 \
    --intervention_key space \
    --num_episodes 10 \
    --horizon 800
```

**Key Arguments:**
- `--task`: Environment task name
- `--checkpoint`: Path to pre-trained robomimic checkpoint
- `--dataset_file`: Output path for recorded demonstrations
- `--intervention_key`: Key to press to take control (default: space)
- `--num_episodes`: Number of human-controlled episodes to collect
- `--horizon`: Maximum steps per rollout

**Workflow:**
1. Policy controls the robot
2. Press intervention key to take control
3. Teleoperate until episode success
4. Episode is automatically recorded
5. Returns to policy control after reset

### 2. dagger_finetune.py

Fine-tunes a policy on the extended dataset (original + DAgger demos).

**Usage:**
```bash
python dagger/scripts/dagger_finetune.py \
    --checkpoint logs/robomimic/.../model_epoch_60.pth \
    --original_dataset datasets/original.hdf5 \
    --new_dataset datasets/dagger_demos.hdf5 \
    --output_dir logs/robomimic/dagger_finetuned \
    --epochs 50
```

**Key Arguments:**
- `--checkpoint`: Path to pre-trained checkpoint
- `--original_dataset`: Original training dataset
- `--new_dataset`: New DAgger demonstrations
- `--output_dir`: Directory for fine-tuned checkpoints
- `--epochs`: Number of fine-tuning epochs
- `--batch_size`: Optional batch size override
- `--learning_rate`: Optional learning rate override

## DAgger Loop Workflow

1. **Collect initial demonstrations** (using record_demos.py)
2. **Train initial policy** (using robomimic/train.py)
3. **Collect DAgger demos** (using dagger_collect.py)
4. **Fine-tune policy** (using dagger_finetune.py)
5. **Evaluate** (using robomimic/play.py)
6. Repeat steps 3-5 until satisfied

## Example Complete Workflow

```bash
# 1. Initial training
python scripts/robomimic/train.py --task ILBL-Go1-Mimic-Box-v0 --algo bc_rnn

# 2. Collect DAgger demonstrations
python dagger/scripts/dagger_collect.py \
    --task ILBL-Go1-Mimic-Box-v0 \
    --checkpoint logs/robomimic/.../model_epoch_60.pth \
    --dataset_file datasets/dagger_round1.hdf5 \
    --num_episodes 10

# 3. Fine-tune on extended dataset
python dagger/scripts/dagger_finetune.py \
    --checkpoint logs/robomimic/.../model_epoch_60.pth \
    --original_dataset datasets/original.hdf5 \
    --new_dataset datasets/dagger_round1.hdf5 \
    --output_dir logs/robomimic/dagger_round1 \
    --epochs 50

# 4. Evaluate fine-tuned policy
python scripts/robomimic/play.py \
    --task ILBL-Go1-Mimic-Box-v0 \
    --checkpoint logs/robomimic/dagger_round1/.../model_epoch_50.pth \
    --num_rollouts 10
```

## Notes

- The collector uses pynput for global keyboard listening
- Only successful human-controlled episodes are recorded
- The fine-tuning script merges datasets before training
- Checkpoints are saved periodically during fine-tuning
```

**Step 2: Commit**

```bash
git add dagger/README.md
git commit -m "docs(dagger): add README with usage instructions"
```

---

## Summary

This implementation provides:

1. **dagger_collect.py**: Interactive data collection with human intervention
2. **dagger_finetune.py**: Fine-tuning on extended dataset
3. **Full documentation** in README.md

The DAgger algorithm is now ready to use for iterative policy improvement through human demonstration.
