from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab.envs.mdp.events import reset_root_state_uniform

def reset_root_state_uniform_and_terrian(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Finish `reset_root_state_uniform` and reset the terrain.
    This function decorates the `reset_root_state_uniform` function by resetting the terrain after resetting the root state.
    The randomization is a hack, got from part of IsaacLab/source/isaaclab/isaaclab/terrains/terrain_generator.py, which is originally
    used for curriculum learning on terrains.
    """

    # Randomize terrain origin for diverse data collection
    # env.scene.terrain.env_origins[0] = random selection from terrain_origins grid
    terrain_rows = env.scene.terrain.cfg.terrain_generator.num_rows
    terrain_cols = env.scene.terrain.cfg.terrain_generator.num_cols

    random_row = torch.randint(0, terrain_rows, (len(env_ids),) )
    random_col = torch.randint(0, terrain_cols, (len(env_ids),) )
    
    env.scene.terrain.terrain_levels[env_ids] = random_row
    env.scene.terrain.terrain_types[env_ids] = random_col

    env.scene.terrain.env_origins[env_ids] = \
        env.scene.terrain.terrain_origins[
            env.scene.terrain.terrain_levels[env_ids], 
            env.scene.terrain.terrain_types[env_ids]
        ]
    
    # reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg)
    reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg)

