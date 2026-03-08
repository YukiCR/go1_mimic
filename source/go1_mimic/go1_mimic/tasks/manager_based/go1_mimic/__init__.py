# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="ILBL-Go1-Nav-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go1_mimic_env_cfg:NavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NavigationEnvPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="ILBL-Go1-Mimic-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go1_mimic_env_cfg:Go1MimicFlatEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
)

gym.register(
    id="ILBL-Go1-Mimic-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go1_mimic_env_cfg:Go1MimicRoughEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_200.json",
    },
)