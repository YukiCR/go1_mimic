# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

import numpy as np

from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg, ContactSensorCfg, RayCasterCfg, RayCaster, patterns
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, MeshRepeatedBoxesTerrainCfg, MeshRepeatedCylindersTerrainCfg, MeshRepeatedPyramidsTerrainCfg, FlatPatchSamplingCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import go1_mimic.tasks.manager_based.go1_mimic.mdp as mdp # extends the isaaclab.envs.mdp with custom functions

##
# Pre-defined configs
##

from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG # base config of go1 robot

# terrian config with repeated boxes, cylinders and pyramids, used for training with repeated obstacles
REPEATED_OBS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(15.0, 15.0),
    border_width=3.0,
    num_rows=15,
    num_cols=6,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "repeated_boxes": MeshRepeatedBoxesTerrainCfg(
                            proportion= 0.33,
                            object_params_start= MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                                                        num_objects=20,
                                                        height=1.5,
                                                        size=(0.6,0.6),
                                                        max_yx_angle=30.0
                                                    ),
                            object_params_end= MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                                                        num_objects=35,
                                                        height=1.5,
                                                        size=(1.2,1.2),
                                                        max_yx_angle=30.0
                                                    ),
                            platform_width=1.0,
                            platform_height=0.0,
                            rel_height_noise=(0.8, 1.5),
                            flat_patch_sampling={
                                "target": FlatPatchSamplingCfg(
                                            num_patches=20,
                                            patch_radius=[0.25, 0.5, 0.75, 1.0],
                                            max_height_diff=0.01,
                                            x_range=(-10, 10), 
                                            y_range=(-10, 10),
                                            z_range=(-0.01, 0.01)
                                        )
                                }
        ),
        "repeated_cylinders": MeshRepeatedCylindersTerrainCfg(
                            proportion= 0.33,
                            object_params_start= MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                                                        num_objects=25,
                                                        height=2.0,
                                                        radius=0.3,
                                                        max_yx_angle=30.0
                                                    ),
                            object_params_end= MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                                                        num_objects=40,
                                                        height=2.0,
                                                        radius=0.6,
                                                        max_yx_angle=30.0
                                                    ),
                            platform_width=1.0,
                            platform_height=0.0,
                            rel_height_noise=(0.5, 2.0),
                                flat_patch_sampling={
                                "target": FlatPatchSamplingCfg(
                                            num_patches=20,
                                            patch_radius=[0.25, 0.5, 0.75, 1.0],
                                            max_height_diff=0.01,
                                            x_range=(-10, 10), 
                                            y_range=(-10, 10),
                                            z_range=(-0.01, 0.01)
                                        )
                                }
        ),
        "repeated_pyramids": MeshRepeatedPyramidsTerrainCfg(
                            proportion= 0.33,
                            object_params_start= MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                                                        num_objects=20,
                                                        height=1.5,
                                                        radius=0.6,
                                                        max_yx_angle=30.0
                                                    ),
                            object_params_end= MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                                                        num_objects=35,
                                                        height=1.5,
                                                        radius=1.2,
                                                        max_yx_angle=30.0
                                                    ),
                            platform_width=1.0,
                            platform_height=0.0,
                            rel_height_noise=(0.8, 1.5),
                            flat_patch_sampling={
                                "target": FlatPatchSamplingCfg(
                                            num_patches=20,
                                            patch_radius=[0.25, 0.5, 0.75, 1.0],
                                            max_height_diff=0.01,
                                            x_range=(-10, 10), 
                                            y_range=(-10, 10),
                                            z_range=(-0.01, 0.01)
                                        )
                                }
        ),
    },  
)

# low-level observation config, used for pre-trained policy action
@configclass
class LowLevelObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()

##
# Scene definition
##


@configclass
class Go1MimicSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane, start with flat plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", 
        terrain_type="plane",
        terrain_generator=None,
        debug_vis=False
    )

    # robot
    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    lidar_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0167)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels = 64,
            vertical_fov_range=(0.0,0.0),
            horizontal_fov_range=(-180.0,180.0),
            horizontal_res=5.0,
        ),
        max_distance= 5.0,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk/front_cam",
        offset=CameraCfg.OffsetCfg(pos=(0.2785, 0.0125, 0.0167), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        data_types=[
                    "rgb",    # uncomment to enable rgb image
                    "depth"
                ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)
        ),
        width=64,
        height=64,
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        track_air_time=True
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path="source/go1_mimic/go1_mimic/tasks/manager_based/go1_mimic/policy/policy.pt",
        # defined in source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
        low_level_decimation=4, 
        # defined in source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go1/rough_env_cfg.py
        low_level_actions=mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True),
        # defined in source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go1/flat_env_cfg.py
        low_level_observations=LowLevelObservationsCfg().policy,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
    )


##
# Environment configuration
##


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: Go1MimicSceneCfg = Go1MimicSceneCfg(num_envs=256, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""

        self.decimation = 4 * 10
        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
        self.sim.device = 'cuda:0'

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if hasattr(self.scene, "height_scanner") and self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt  # 50 Hz
        if hasattr(self.scene, "camera") and self.scene.camera is not None:
            self.scene.camera.update_period = self.decimation * self.sim.dt  # 50 Hz
        if hasattr(self.scene, "contact_forces") and self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.decimation * self.sim.dt  # 50 Hz
        if hasattr(self.scene, "lidar_scanner") and self.scene.lidar_scanner is not None:
            self.scene.lidar_scanner.update_period  = self.decimation * self.sim.dt * 5  # 10 Hz

class NavigationEnvCfg_PLAY(NavigationEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


##
# IL Environment configuration
##

# ==== flat terrain environment configuration ====
@configclass
class MimicTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},
    )
    # new term
    success = DoneTerm(
        func=mdp.reached_target,
        params={"command_name": "pose_command", "threshold": (0.8, 0.6)},
    )


class Go1MimicFlatEnvCfg(NavigationEnvCfg):
    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground", 
            terrain_type="plane",
            terrain_generator=None,
            debug_vis=False
        )

        # add success termination
        self.terminations = MimicTerminationsCfg()

        # lengthen the command resampling time range
        self.commands.pose_command.resampling_time_range = (16,16)

        # update episode length accordingly
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]


# ==== rough terrain environment configuration ====
@configclass
class MimicTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},
    )
    # new term
    success = DoneTerm(
        func=mdp.reached_distance_target,
        params={"command_name": "pose_command", "threshold": 0.8},
    )

@configclass
class RoughEventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform_and_terrian,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

@configclass
class RoughCommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.TerrainBasedPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.TerrainBasedPose2dCommandCfg.Ranges(heading=(-math.pi, math.pi)),
    )


def sphere_distance(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, radius: float) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # sensor.data.pos_w: [N, 3], sensor.data.ray_hits_w: [N, B, 3]
    pos = sensor.data.pos_w[:, None, :3]               # [N, 1, 3]
    hits = sensor.data.ray_hits_w[..., :3]             # [N, B, 3]
    # compute per-beam distance and take minimum per environment
    dists = torch.norm(hits - pos, dim=-1)             # [N, B]
    min_dists, _ = dists.min(dim=1, keepdim=True)      # [N, 1]
    return min_dists - radius

@configclass
class VisuoObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        # added imgae observations
        rgb_image = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb"}
        )
        depth_image = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "depth"}
        )
        # added distance observation
        sphere_distance = ObsTerm(
            func=sphere_distance,
            params={"sensor_cfg": SceneEntityCfg("lidar_scanner"), "radius": 0.50},
        )

        def __post_init__(self):
            self.enable_corruption = True  # can add disturbance to observation
            self.concatenate_terms = False  # keep terms separate
    
    # observation groups
    policy: PolicyCfg = PolicyCfg()





class Go1MimicRoughEnvCfg(NavigationEnvCfg):
    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground", 
            terrain_type="generator",
            terrain_generator=REPEATED_OBS_TERRAINS_CFG,
            debug_vis=False
        )

        # use the new observation config with vision
        self.observations = VisuoObservationsCfg()  

        # override the reset event to randomize the terrain as well
        self.events.reset_base = RoughEventCfg().reset_base

        # add success termination
        self.terminations = MimicTerminationsCfg()

        # override the command generator to be terrain-based
        self.commands.pose_command = RoughCommandsCfg().pose_command

        # lengthen the command resampling time range
        self.commands.pose_command.resampling_time_range = (25,25)

        # update episode length accordingly
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        self.image_obs_list = ["rgb_image"]