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

def reset_root_state_indoor(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state by sampling a random valid pose from indoor terrain flat patches.

    This function samples a random valid pose from the pre-sampled flat patches in the indoor
    environment (stored under "init_pos") and sets the root state of the asset to this position.
    The function also samples random velocities from the given ranges.

    Unlike reset_root_state_from_terrain which is designed for generated terrains with multiple
    tiles/levels, this function is for USD-imported indoor scenes with a single continuous space.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to reset.
        pose_range: A dictionary of pose ranges for orientation randomization.
            Keys can be ``roll``, ``pitch``, ``yaw``. Position is sampled from flat patches.
        velocity_range: A dictionary of velocity ranges for each axis and rotation.
        asset_cfg: The asset configuration. Defaults to SceneEntityCfg("robot").

    Raises:
        ValueError: If the terrain does not have valid flat patches under the key "init_pos".
    """
    # extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # obtain flat patches for init positions
    valid_positions: torch.Tensor = terrain.flat_patches.get("init_pos")
    if valid_positions is None:
        raise ValueError(
            "The event term 'reset_root_state_indoor' requires valid flat patches under 'init_pos'."
            f" Found: {list(terrain.flat_patches.keys())}"
        )

    # sample random valid poses
    # valid_positions shape: (1, 1, N, 3) for USD terrain (single tile)
    num_patches = valid_positions.shape[2]
    ids = torch.randint(0, num_patches, size=(len(env_ids),), device=env.device)
    # For single-tile USD terrain, always use (0, 0) for level/type
    positions = valid_positions[0, 0, ids]
    # Add default root height offset
    positions[:, 2] += asset.data.default_root_state[env_ids, 2]

    # sample random orientations
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    # convert to quaternions
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = asset.data.default_root_state[env_ids, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


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

