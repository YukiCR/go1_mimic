The core task to implement this time is to add an depth_latent observation term in the issac env config.

+ background:
  + project: this project is a isaaclab project, defining the sim env and the policy learing settings
  + isaac environment defination: the core dir deifining the env is @source/go1_mimic/go1_mimic/tasks/manager_based/go1_mimic , the core file defining the env is @source/go1_mimic/go1_mimic/tasks/manager_based/go1_mimic/go1_mimic_env_cfg.py
  + observation: the whole env is defined in a MDP fashion, the observation config to improve is line 597-631 of @source/go1_mimic/go1_mimic/tasks/manager_based/go1_mimic/go1_mimic_env_cfg.py
  + modifier: modifier serves as a post process for observations, refer to the following files to understand how modifier works and how to develop a modifier:
    + @/home/chengrui/IsaacLab/source/isaaclab/isaaclab/utils/modifiers/modifier_base.py
    + @/home/chengrui/IsaacLab/source/isaaclab/isaaclab/utils/modifiers/modifier_cfg.py
    + @/home/chengrui/IsaacLab/source/isaaclab/isaaclab/utils/modifiers/modifier.py
    + line 146-192 @/home/chengrui/IsaacLab/source/isaaclab/isaaclab/managers/manager_term_cfg.py 
    + @source/go1_mimic/go1_mimic/tasks/manager_based/go1_mimic/go1_mimic_env_cfg.py line 629-634
  + checkpoint: using the depth image collected from sim, autoencoder is trained and saved in @source/go1_mimic/go1_mimic/tasks/manager_based/go1_mimic/pretrained_encoder read the README in this dir

+ task: add a depth_latent term in the observation, where the latent is encoded by the pretrained autoencoder, using the input depth data. To finish this task, we:
  + implement a modifier following the isaaclab pattern, in the class, we load the checkpoint via jit, then given a input data shaped [B, 64, 64,1], we 
    + (1) normalize the data (refer to the following code)
    ```python
      transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Normalize(mean=[1.77], std=[2.58]),
    ])
      # Convert to tensor [H, W, C] -> [C, H, W], then apply transforms
      img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
      img_tensor = transform(img_tensor).unsqueeze(0)

    ```
    + (2) call the model to forward to get latent shaped [B, 64], where 64 is the length of the latent vector. If B is larger than the max batch size 64, process them batch-by-batch
    + (3) return the value
  + finish line 629-634 with correct modifier

+ subtasks:
  + understand: first understand the project, especially how modifier works
  + modifier implementation：implement the modifier 
  + combinaition aiding: finish the observation term, help the user to tune

+ decisions:  
  + use conda env env_isaaclab
  + use superpowers skills
  + develop in git branch feature_modifier