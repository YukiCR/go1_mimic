# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for the indoor navigation environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def base_heading(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root heading (yaw angle) in the simulation world frame.

    The heading is the yaw angle of the robot's base frame, computed as the angle
    between the world x-axis and the robot's forward direction (x-axis in base frame).
    The angle is in radians and ranges from -pi to pi.

    Args:
        env: The environment instance.
        asset_cfg: The asset configuration. Defaults to SceneEntityCfg("robot").

    Returns:
        Tensor of shape (num_envs, 1) containing the heading angle in radians.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.heading_w.unsqueeze(-1)
