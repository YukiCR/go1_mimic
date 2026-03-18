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

            # Get new demos (filter out empty demos with no actions)
            new_demos = []
            for k in f_new["data"].keys():
                if k.startswith("demo_"):
                    demo_group = f_new[f"data/{k}"]
                    if "actions" in demo_group and demo_group["actions"].shape[0] > 0:
                        new_demos.append(k)
                    else:
                        print(f"[WARNING] Skipping empty demo '{k}' (no actions or zero-length)")

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

    # Unlock config to allow modifications
    config.unlock()

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

    # Setup log file redirection (similar to train.py)
    if config.experiment.logging.terminal_output_to_txt:
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

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
