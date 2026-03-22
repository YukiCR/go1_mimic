The final goal of current task is to realize the `DAgger` algorithm in imitation learning, enabling extending the dataset based on the human supervision.

#### DAgger: 
DAgger helps to alleviate the mismatch between imitation policy and demonstartion data. Key points include:
+ pre-train on collected data
+ rollout pre-trained model
+ expert intervention for demonstration on rollout observations
+ extend the original data with newly collected demonstration data, then further train the model using the extended data
+ repeat the loop until the model is satisfying

Read the reference link for further understanding : https://imitation.readthedocs.io/en/latest/algorithms/dagger.html
The original papaer in at https://arxiv.org/pdf/1011.0686
Also check the internet until you fully captured the Dagger algo.

#### Backgound of this project:
+ Isaaclab: This project is a isaaclab project, aiming to realize the imitation learning of navigaiton task of quadrupted robot. Refer to  https://isaac-sim.github.io/IsaacLab/main/index.html for isaaclab documentation of we encounter any isaaclab realated problems. These files should be carefully read to understand the isaaclab application in this project:
  + @scripts/robomimic/play.py
  + @scripts/robomimic/robust_eval.py
  + @scripts/robomimic/train.py  
  + @scripts/tools/record_demos.py
  + @scripts/tools/replay_demos.py
Also understand the details of the following mechanism:
  + How data recording is realized (via issaclab recorder manager)
  + How teleoperation is realized (via teleop_interface)
  + How to realize a dagger interactive traj collector script enabling:
    + pre-trained policy rollout
    + real-time human intervention for further demonstration
    + recording the human demonstration as dataset augmentation
If reading source code is needed, see the issaclab code in @/home/chengrui/IsaacLab
+ robomimic: The imitation learning is currently done with `robomimic`. Refer to https://robomimic.github.io/docs/index.html for the documentation. It is recommanded to scan @/home/chengrui/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/robomimic/ dir to understand the robomimic project. Spectal attention should be paid to the training pipeline of the model, cause Dagger requires to finetune a model using extended dataset. So these questions should be kept in mind: 
  + how to load a pretrained model, 
  + how to setup the finetuning pipeline.
+ go1_mimic: go1_mimic is the current issalab env definiation. The env is defined in @source/go1_mimic/go1_mimic/tasks/manager_based/go1_mimic , the core file is @source/go1_mimic/go1_mimic/tasks/manager_based/go1_mimic/go1_mimic_env_cfg.py , understand the following aspects of the env:
  + the observation
  + the action
  + the termination
  + the reset
  + the terrian setting  
  The env we currently focus on is `Go1MimicBoxEnvCfg`, which is the simpliest env to validate the IL pipeline.
+ data example:
  + the checkpoint example:  @logs/robomimic/ILBL-Go1-Mimic-Box-v0/bc_rnn_go1_nav_lader_box/20260315145512/models/model_epoch_60.pth
  + the dataset example: @datasets/box_dataset_0.hdf5

#### Goal
The goal is to extend current one-time learning design to DAgger fashion IL. What we need is:
+ a interactive traj collector script, which:
  + loads the checkpoint
  + enabling expert demonstration (human teleoperation) when rolling out the policy
  + collects the demonstration to a dataset file
+ a fintuning script, which:
  + load a checkpoint
  + finetune a model checkpoint with given dataset

#### Note:
+ understanding goes first: The coding amout is small, yet we have to understand the current modules (isaaclab and robomimic) to ensure we code it right. Read enough to ensure we understand how curent code works, only code after you and the user understand and have concensus on the current pattern and what we finally need.
+ Take note if needed: after reading and understaning a module, feel free to take notes in @dagger/notes
+ implement code: write code in @dagger/scripts




