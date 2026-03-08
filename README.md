# go1_mimic

The isaaclab env enabling RL/IL of the navigation task of unitree go1 robot.
Modified from the `Isaac-Navigation-Flat-Anymal-C-v0` environment of isaaclab.

![Description of the image](imgs/capture.png)

## Requirements
This env is developed and tested with:
+ isaacsim 4.5
+ isaaclab, `main` branch, commit `d13cb0b6043db6ae9b2efc3ab1ac64f7a77ed3ed` , date Sun Dec 28 02:27:34 2025 +0100, Fixes backward compatibility to IsaacSim 4.5 for new stage utils (#4230)
+ ubuntu 20.04 LTS

Other versions/OSs are very likely to work as well.

## Run
1. follow [Template_README.md](Template_README.md) to install this env to your isaaclab python env via `pip install`
2. RL
   1. launch with
   ```bash
    python scripts/rsl_rl/train.py --task ILBL-Go1-Nav-Flat-v0 --enable_cameras
   ```
3. IL  
   Install `robomimic` following isaaclab's instarucitions first. If you installed all modules on isaaclab installation, robomimic has then been already installed.

   1. collect data with
   ```bash
   python scripts/tools/record_demos.py --task ILBL-Go1-Mimic-Rough-v0 --dataset_file ./dataset/dataset.hdf5 --enable_cameras 
   ```
   use arrow keys and Z, X key to teleoperate with keyboard  

   2. train with 
   ```bash
    python scripts/robomimic/train.py --task ILBL-Go1-Mimic-Rough-v0 --algo bc  --dataset ./dataset/dataset.hdf5
   ```
   3. view trained policy with
   ```bash
    python scripts/robomimic/play.py --enable_cameras --task ILBL-Go1-Mimic-Rough-v0 --num_rollouts 20 --horizon 100 --checkpoint logs/robomimic/ILBL-Go1-Mimic-Rough-v0/bc_rnn_image_go1_nav/XXXXXX/models/XXXXXX.pth
   ```