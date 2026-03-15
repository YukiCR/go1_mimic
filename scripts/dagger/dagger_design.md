# DAgger Implementation Design

## Overview
Dataset Aggregation (DAgger) algorithm implementation for IsaacLab/robomimic imitation learning pipeline.

## Algorithm Flow

```
Loop:
1. Rollout pre-trained policy in environment
2. Human monitors the rollout
3. Human presses INTERVENTION_KEY to take control
4. Human teleoperates the robot until episode success
5. Episode is recorded and exported to HDF5
6. Environment resets
7. New dataset = Original dataset + New demonstrations
8. Fine-tune policy on extended dataset
9. Repeat from step 1 with improved policy
```

## Components

### 1. Interactive Trajectory Collector (`dagger_collect.py`)

**Purpose**: Rollout policy with human intervention capability

**Key Features**:
- Loads pre-trained robomimic checkpoint
- Runs policy inference loop
- Monitors for human intervention (keyboard key)
- Switches control between policy and human teleop
- Records human-controlled episodes via IsaacLab RecorderManager
- Exports successful demonstrations to HDF5

**Control States**:
- `POLICY`: Policy controls the robot (default)
- `HUMAN`: Human teleop controls the robot (after intervention)

**State Transitions**:
```
POLICY --[INTERVENTION_KEY]--> HUMAN
HUMAN --[success + reset]--> POLICY
```

**Intervention Mechanism**:
- Non-blocking keyboard input check each step
- When intervention triggered:
  - Stop policy inference
  - Switch to teleop_interface for actions
  - Continue recording the same episode
- Episode ends on success condition
- Export demonstration
- Reset and return to POLICY state

### 2. Fine-tuning Script (`dagger_finetune.py`)

**Purpose**: Fine-tune policy on extended dataset

**Key Features**:
- Loads pre-trained checkpoint
- Combines original dataset with new demonstrations
- Runs robomimic training loop
- Saves fine-tuned checkpoints

**Dataset Handling**:
- Option A: Merge HDF5 files (original + new demos)
- Option B: Use dataset directory (robomimic supports multiple dataset paths)

## Implementation Details

### Interactive Collector Architecture

```python
class DAggerCollector:
    states = [POLICY, HUMAN]

    def __init__(self, checkpoint_path, dataset_output_path):
        self.policy = load_policy(checkpoint_path)
        self.teleop = setup_teleop_device()
        self.env = create_env_with_recorder(dataset_output_path)
        self.state = POLICY

    def run(self):
        while running:
            if self.state == POLICY:
                action = self.policy(obs)
                if intervention_key_pressed():
                    self.state = HUMAN
                    continue
            else:  # HUMAN
                action = self.teleop.advance()

            obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:  # Success or failure
                if self.state == HUMAN:
                    # Episode was human-controlled, mark as successful
                    env.recorder_manager.set_success_to_episodes([0], True)
                    env.recorder_manager.export_episodes([0])

                self.env.reset()
                self.state = POLICY  # Return to policy control
```

### Key Integration Points

**IsaacLab Integration**:
- Uses `ActionStateRecorderManagerCfg` for recording
- Recorder manager automatically captures states, actions, observations
- Success marking via `recorder_manager.set_success_to_episodes()`
- Export via `recorder_manager.export_episodes()`

**Robomimic Integration**:
- Policy loading via `FileUtils.policy_from_checkpoint()`
- Observation preprocessing (same as play.py)
- Action normalization support

**Teleop Integration**:
- Reuses Se2Keyboard from record_demos.py
- Non-blocking action retrieval
- Callback system for intervention key

## File Structure

```
dagger/
├── notes/
│   └── dagger_design.md          # This document
├── scripts/
│   ├── dagger_collect.py         # Interactive trajectory collector
│   ├── dagger_finetune.py        # Fine-tuning script
│   └── dagger_loop.py            # (Optional) Full DAgger loop automation
└── README.md                     # Usage instructions
```

## CLI Design

### dagger_collect.py
```bash
python dagger/scripts/dagger_collect.py \
    --task ILBL-Go1-Mimic-Box-v0 \
    --checkpoint logs/robomimic/.../model_epoch_60.pth \
    --dataset_file datasets/dagger_new_demos.hdf5 \
    --intervention_key "space" \
    --num_episodes 10 \
    --horizon 800
```

### dagger_finetune.py
```bash
python dagger/scripts/dagger_finetune.py \
    --checkpoint logs/robomimic/.../model_epoch_60.pth \
    --original_dataset datasets/original.hdf5 \
    --new_dataset datasets/dagger_new_demos.hdf5 \
    --output_dir logs/robomimic/dagger_finetuned \
    --epochs 50
```

## Design Decisions

1. **Single-episode recording**: Each human intervention creates one demonstration
2. **Success-based termination**: Episode ends only on success condition
3. **HDF5 export**: Compatible with existing robomimic training pipeline
4. **Modular design**: Collection and fine-tuning are separate scripts
5. **Reuses existing code**: Leverages play.py, record_demos.py patterns

## Future Extensions

- Automatic DAgger loop (run collection + fine-tuning iteratively)
- Intervention heuristics (auto-intervene on policy uncertainty)
- Partial episode recording (intervene, correct, return to policy)
- Multi-environment parallel collection
