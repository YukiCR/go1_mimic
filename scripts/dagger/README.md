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

### 1. pure_dagger_collect.py (Recommended)

Pure DAgger data collection with implicit intervention detection. This is the recommended script for DAgger as it closely follows the original algorithm.

**Usage:**
```bash
python scripts/dagger/pure_dagger_collect.py \
    --task ILBL-Go1-Mimic-Rough-v0 \
    --checkpoint logs/robomimic/ILBL-Go1-Mimic-Rough-v0/bc_rnn_go1_nav_latent_lader/20260316022213/models/model_epoch_560.pth \
    --dataset_file datasets/pure_dagger_demos.hdf5 \
    --num_segments 200 \
    --horizon 100 \
    --debounce_steps 2 \
    --min_segment_length 5 \
    --enable_cameras
```

**Key Arguments:**
- `--task`: Environment task name
- `--checkpoint`: Path to pre-trained robomimic checkpoint
- `--dataset_file`: Output path for recorded demonstrations
- `--num_segments`: Number of intervention segments to collect (default: 100)
- `--horizon`: Maximum steps per rollout (default: 50)
- `--debounce_steps`: Consecutive non-zero actions to trigger intervention (default: 2)
- `--min_segment_length`: Minimum steps for a valid segment (default: 5)
- `--norm_factor_min`: Optional minimum normalization factor for action unnormalization
- `--norm_factor_max`: Optional maximum normalization factor for action unnormalization

**Workflow:**
1. Policy controls the robot
2. When you press teleop keys, intervention is automatically detected (no key to press)
3. After `debounce_steps` consecutive non-zero actions, switches to human mode
4. Human actions are visualized as a **red arrow** above the robot
5. When you stop pressing keys, the segment is exported and returns to policy mode
6. Multiple segments can be collected within a single rollout

**Key Bindings for Teleop:**
- Arrow Up / Numpad 8: Move forward
- Arrow Down / Numpad 2: Move backward
- Arrow Left / Numpad 6: Move left
- Arrow Right / Numpad 4: Move right
- Z / Numpad 7: Rotate left
- X / Numpad 9: Rotate right
- L: Reset all commands

**Why this is "Pure" DAgger:**
- Records `(s_policy, a_human)` pairs - states visited by the policy with human corrective actions
- No explicit intervention key needed - detects human input implicitly
- Policy RNN state is maintained throughout (even during human control)
- Multiple short segments per rollout (more efficient data collection)

### 2. dagger_collect.py

Alternative collector with explicit intervention key (original implementation).

**Usage:**
```bash
python scripts/dagger/dagger_collect.py \
    --task ILBL-Go1-Mimic-Box-v0 \
    --checkpoint logs/robomimic/.../model_epoch_60.pth \
    --dataset_file datasets/dagger_demos.hdf5 \
    --intervention_key P \
    --num_episodes 10 \
    --horizon 800
```

**Key Arguments:**
- `--task`: Environment task name
- `--checkpoint`: Path to pre-trained robomimic checkpoint
- `--dataset_file`: Output path for recorded demonstrations
- `--intervention_key`: Key to press to take control (default: SPACE). **Note:** Do not use SPACE as it conflicts with IsaacSim's pause hotkey. Use "P" or other keys.
- `--num_episodes`: Number of human-controlled episodes to collect
- `--horizon`: Maximum steps per rollout

**Workflow:**
1. Policy controls the robot
2. Press intervention key (e.g., "P") to take control
3. Teleoperate using arrow keys / numpad until episode success
4. Episode is automatically recorded and exported
5. Returns to policy control after reset

### 3. dagger_finetune.py

Fine-tunes a policy on the extended dataset (original + DAgger demos).

**Usage:**
```bash
python scripts/dagger/dagger_finetune.py \
    --checkpoint logs/robomimic/ILBL-Go1-Mimic-Rough-v0/bc_rnn_go1_nav_latent_lader/20260316022213/models/model_epoch_560.pth \
    --original_dataset datasets/latent/pretrain/dataset_latent_straight_merged.hdf5 \
    --new_dataset datasets/latent/dagger/latent_test.hdf5 \
    --output_dir logs/robomimic/ILBL-Go1-Mimic-Rough-v0/bc_rnn_go1_nav_latent_lader/dagger_finetuned_0 \
    --learning_rate 0.0001
    --epochs 10
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

1. **Collect initial demonstrations** (using `scripts/tools/record_demos.py`)
2. **Train initial policy** (using `scripts/robomimic/train.py`)
3. **Collect DAgger demos** (using `scripts/dagger/pure_dagger_collect.py`)
4. **Fine-tune policy** (using `scripts/dagger/dagger_finetune.py`)
5. **Evaluate** (using `scripts/robomimic/play.py`)
6. Repeat steps 3-5 until satisfied

## Example Complete Workflow

```bash
# 1. Initial training
python scripts/robomimic/train.py --task ILBL-Go1-Mimic-Box-v0 --algo bc_rnn

# 2. Collect DAgger demonstrations (pure_dagger_collect.py recommended)
python scripts/dagger/pure_dagger_collect.py \
    --task ILBL-Go1-Mimic-Box-v0 \
    --checkpoint logs/robomimic/.../model_epoch_60.pth \
    --dataset_file datasets/dagger_round1.hdf5 \
    --num_segments 100 \
    --horizon 50

# 3. Fine-tune on extended dataset
python scripts/dagger/dagger_finetune.py \
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

## Important Notes

- **pure_dagger_collect.py is recommended** over `dagger_collect.py` as it implements true DAgger with implicit intervention
- **Red arrow visualization**: In `pure_dagger_collect.py`, a red arrow appears above the robot when you're controlling it
- **Only successful segments recorded**: Segments shorter than `min_segment_length` are discarded
- **Multiple segments per rollout**: `pure_dagger_collect.py` can collect multiple segments within a single episode

## Troubleshooting

**Freeze after intervention (dagger_collect.py only):**
- Check if your intervention key conflicts with IsaacSim hotkeys (SPACE, F, etc.)
- Try using a different key like "P", "I", or "H"
- Consider using `pure_dagger_collect.py` instead which doesn't use intervention keys

**No demos recorded:**
- Ensure the success condition is being met (check environment logs)
- Verify the recorder manager is properly configured
- Check that `dataset_file` path is writable
- For `pure_dagger_collect.py`, ensure segments are longer than `min_segment_length`

**Policy not loading:**
- Verify checkpoint path is correct
- Check that the checkpoint is compatible with the task/environment

**Device mismatch errors:**
- Ensure CUDA is available or run on CPU
- The scripts automatically handle device selection
