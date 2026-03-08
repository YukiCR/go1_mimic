from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""


def reached_target(env: ManagerBasedRLEnv, command_name: str, threshold: tuple) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    heading_b = command[:, 3]
    return distance <= threshold[0] and heading_b.abs() <= threshold[1]

def reached_distance_target(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return distance <= threshold