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

**Key Bindings for Teleop:**
- Arrow Up / Numpad 8: Move forward
- Arrow Down / Numpad 2: Move backward
- Arrow Left / Numpad 6: Move left
- Arrow Right / Numpad 4: Move right
- Z / Numpad 7: Rotate left
- X / Numpad 9: Rotate right
- L: Reset all commands

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

1. **Collect initial demonstrations** (using `scripts/tools/record_demos.py`)
2. **Train initial policy** (using `scripts/robomimic/train.py`)
3. **Collect DAgger demos** (using `dagger/scripts/dagger_collect.py`)
4. **Fine-tune policy** (using `dagger/scripts/dagger_finetune.py`)
5. **Evaluate** (using `scripts/robomimic/play.py`)
6. Repeat steps 3-5 until satisfied

## Example Complete Workflow

```bash
# 1. Initial training
python scripts/robomimic/train.py --task ILBL-Go1-Mimic-Box-v0 --algo bc_rnn

# 2. Collect DAgger demonstrations (use "P" key to intervene, not SPACE)
python dagger/scripts/dagger_collect.py \
    --task ILBL-Go1-Mimic-Box-v0 \
    --checkpoint logs/robomimic/.../model_epoch_60.pth \
    --dataset_file datasets/dagger_round1.hdf5 \
    --intervention_key P \
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

## Important Notes

- **Intervention Key:** Do NOT use "SPACE" as the intervention key - it's already bound to pause simulation in IsaacSim. Use "P" or another unused key.
- **Only successful episodes recorded:** If the human fails to complete the task, the episode is discarded.
- **Episode continuation:** After intervention, the episode continues from the current state (not reset).
- **Return to policy:** After a successful human demonstration and environment reset, control automatically returns to the policy.

## Troubleshooting

**Freeze after intervention:**
- Check if your intervention key conflicts with IsaacSim hotkeys (SPACE, F, etc.)
- Try using a different key like "P", "I", or "H"

**No demos recorded:**
- Ensure the success condition is being met (check environment logs)
- Verify the recorder manager is properly configured
- Check that `dataset_file` path is writable

**Policy not loading:**
- Verify checkpoint path is correct
- Check that the checkpoint is compatible with the task/environment
