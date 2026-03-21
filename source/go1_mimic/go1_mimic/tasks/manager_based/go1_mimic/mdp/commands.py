# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command generators for indoor navigation environments."""

# NOTE: Workaround for IsaacLab raycast_mesh bug (ops.py:123)
#
# The raycast_mesh function has a bug when using return_distance=True with 2D inputs
# of shape (N, 3). At line 63, it stores `shape = ray_starts.shape` which captures
# the original input shape (N, 3). Later at line 123, when return_distance=True,
# it attempts:
#     ray_distance = ray_distance.to(device).view(shape[0], shape[1])
#
# For 2D input (N, 3):
#   - ray_distance has shape (N,) after raycasting (one scalar distance per ray)
#   - view(shape[0], shape[1]) = view(N, 3) tries to reshape N elements into N*3 slots
#   - This fails with: "shape '[N, 3]' is invalid for input of size N"
#
# The function works correctly only for 3D inputs (B, N, 3) where shape[0]=B and
# shape[1]=N, producing output shape (B, N).
#
# To use return_distance=True correctly with 2D inputs, the fix would be to change
# line 123 from:
#     ray_distance = ray_distance.to(device).view(shape[0], shape[1])
# to:
#     ray_distance = ray_distance.to(device).view(shape[:-1])
#
# This would correctly produce shape (N,) for 2D input and (B, N) for 3D input.
#
# Until this is fixed upstream, we compute distance manually from ray_hits using
# torch.norm(ray_hits[..., :2] - starts[..., :2], dim=-1).

from __future__ import annotations

from dataclasses import MISSING
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands import UniformPose2dCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformPose2dCommandCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainImporter
from isaaclab.terrains.utils import find_flat_patches
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi
from isaaclab.utils.warp import convert_to_warp_mesh

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class IndoorPose2dCommand(UniformPose2dCommand):
    """Command generator for indoor (USD-imported) environments.

    At construction time the class:

    1. Traverses all mesh prims under the terrain's USD prim path (same pattern used by
       :class:`~isaaclab.sensors.ray_caster.MultiMeshRayCaster`).
    2. Converts them to a single warp mesh in world coordinates.
    3. Calls :func:`~isaaclab.terrains.utils.find_flat_patches` with the provided
       :class:`~isaaclab.terrains.FlatPatchSamplingCfg` to locate obstacle-free floor patches.
    4. Injects the result into ``terrain._terrain_flat_patches["target"]`` (shape ``(1, 1, N, 3)``)
       so no IsaacLab source code needs to be modified.

    At every resample the class uniformly draws one of the pre-sampled patches as the new goal.
    """

    cfg: IndoorPose2dCommandCfg

    def __init__(self, cfg: IndoorPose2dCommandCfg, env: ManagerBasedEnv):
        # Call parent: sets up self.robot, self.pos_command_w, self.heading_command_w, etc.
        super().__init__(cfg, env)

        # Access the terrain importer already loaded by the scene
        self.terrain: TerrainImporter = env.scene["terrain"]

        if "target" not in self.terrain._terrain_flat_patches:
            # ----------------------------------------------------------------
            # Step 1: build a combined warp mesh from the imported USD terrain
            # ----------------------------------------------------------------
            terrain_prim_path = self.terrain.terrain_prim_paths[0]  # e.g. "/World/ground/terrain"
            print(f"[IndoorPose2dCommand] Building warp mesh from '{terrain_prim_path}' …")
            wp_mesh = self._build_warp_mesh_from_prim(terrain_prim_path)

            # ----------------------------------------------------------------
            # Step 2: sample flat patches on the floor using raycasting
            # ----------------------------------------------------------------
            patch_cfg = cfg.flat_patch_sampling
            print(
                f"[IndoorPose2dCommand] Sampling {patch_cfg.num_patches} flat patches "
                f"(patch_radius={patch_cfg.patch_radius}, z_range={patch_cfg.z_range}) …"
            )
            patches = find_flat_patches(
                wp_mesh=wp_mesh,
                num_patches=patch_cfg.num_patches,
                patch_radius=patch_cfg.patch_radius,
                origin=(0.0, 0.0, 0.0),    # world frame; returned patches are also world-frame
                x_range=patch_cfg.x_range,
                y_range=patch_cfg.y_range,
                z_range=patch_cfg.z_range,
                max_height_diff=patch_cfg.max_height_diff,
            )
            print(f"[IndoorPose2dCommand] Sampled {patches.shape[0]} candidate patches.")

            # ----------------------------------------------------------------
            # Step 3: discard outdoor patches with a wall-proximity filter.
            #
            # Root cause of floating goals: find_flat_patches fires rays straight
            # DOWN and z_range cannot distinguish indoor floor from outdoor ground
            # at the same elevation.  We fix this by also shooting 4 HORIZONTAL
            # rays (±x, ±y) from every candidate patch.
            #
            # Key insight: an indoor patch is enclosed — walls exist in every
            # cardinal direction within the building's width.
            # An outdoor patch has at least one direction that is completely open;
            # its ray either hits nothing (returns inf) or hits a surface that is
            # much farther away than any interior wall.
            #
            # Both cases are caught by: hit_distance > indoor_filter_wall_distance.
            # Using a distance threshold is more robust than checking for inf alone
            # because small mesh gaps can let rays slip through to a very far
            # surface (large but finite distance) even for an indoor position.
            # ----------------------------------------------------------------
            patches = self._filter_indoor_patches(patches, wp_mesh, cfg.indoor_filter_wall_distance)
            if patches.shape[0] == 0:
                raise RuntimeError(
                    "[IndoorPose2dCommand] No indoor patches survived the wall-proximity filter. "
                    "Try increasing 'indoor_filter_wall_distance' or widening flat_patch_sampling ranges."
                )
            print(
                f"[IndoorPose2dCommand] {patches.shape[0]} patches kept after indoor filter "
                f"(wall_distance_threshold={cfg.indoor_filter_wall_distance} m)."
            )

            # patches: (N, 3) world-frame coordinates
            # Store as (1, 1, N, 3) to match TerrainBasedPose2dCommand conventions
            self.terrain._terrain_flat_patches["target"] = patches.unsqueeze(0).unsqueeze(0)

        # Keep a fast reference for _resample_command
        self.valid_targets: torch.Tensor = self.terrain.flat_patches["target"]  # (1, 1, N, 3)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_warp_mesh_from_prim(self, prim_path: str):
        """Return a warp mesh covering all mesh geometry under *prim_path*.

        The approach mirrors :class:`~isaaclab.sensors.ray_caster.MultiMeshRayCaster`:

        * Traverse all prims under *prim_path* and collect those of type ``Mesh``
          or a USD primitive shape (Cube, Sphere, Cylinder, Capsule, Cone, Plane).
        * Convert each to a :class:`trimesh.Trimesh` (handles quad-to-triangle conversion).
        * Apply the full world transform so the mesh is in world coordinates.
        * Concatenate and convert to a :class:`warp.Mesh`.
        """
        import numpy as np
        import trimesh
        from pxr import Usd
        import omni.usd
        from isaaclab.utils.mesh import create_trimesh_from_geom_mesh, create_trimesh_from_geom_shape

        PRIMITIVE_MESH_TYPES = {"Cube", "Sphere", "Cylinder", "Capsule", "Cone", "Plane"}

        stage = omni.usd.get_context().get_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            raise RuntimeError(f"USD prim at '{prim_path}' is not valid. Check the terrain prim path.")

        trimesh_meshes: list[trimesh.Trimesh] = []
        for prim in Usd.PrimRange(root_prim):
            prim_type = prim.GetTypeName()
            if prim_type not in PRIMITIVE_MESH_TYPES and prim_type != "Mesh":
                continue
            try:
                mesh: trimesh.Trimesh
                if prim_type == "Mesh":
                    mesh = create_trimesh_from_geom_mesh(prim)
                else:
                    mesh = create_trimesh_from_geom_shape(prim)
            except Exception:
                # Skip prims with missing / unreadable geometry (e.g. USD instances)
                continue

            # Apply world transform so vertices are in world coordinates.
            # omni.usd.get_world_transform_matrix returns a Gf.Matrix4d stored in
            # row-major order; transposing gives the standard column-major 4×4 matrix
            # with rotation in [:3, :3] and translation in [:3, 3].
            world_tf = np.array(omni.usd.get_world_transform_matrix(prim)).T
            tf_4x4 = np.eye(4)
            tf_4x4[:3, :3] = world_tf[:3, :3]   # rotation (+ scale absorbed)
            tf_4x4[:3, 3] = world_tf[:3, 3]     # translation
            mesh.apply_transform(tf_4x4)

            trimesh_meshes.append(mesh)

        if not trimesh_meshes:
            raise RuntimeError(
                f"No mesh geometry found under USD prim '{prim_path}'. "
                "Verify the terrain prim path and that the USD file was loaded successfully."
            )

        combined = trimesh.util.concatenate(trimesh_meshes)
        return convert_to_warp_mesh(combined.vertices, combined.faces, device=self.device)

    def _filter_indoor_patches(
        self,
        patches: torch.Tensor,
        wp_mesh,
        max_wall_distance: float,
    ) -> torch.Tensor:
        """Return only the patches that are enclosed by walls in all 4 cardinal directions.

        For each candidate patch we fire 4 horizontal rays (±x, ±y) at robot-torso
        height (+0.5 m above the patch).  A patch is considered **indoor** when
        every ray hits a surface within *max_wall_distance* metres.

        Outdoor patches fail because at least one ray either:
        * hits nothing (distance = inf), or
        * hits a surface that is farther away than any interior wall
          (distance > max_wall_distance).

        Using a **distance threshold** rather than an exact inf check is intentional:
        small holes or thin gaps in the warehouse mesh can let a ray slip through to
        a distant surface instead of returning inf, which would fool an inf-only test.
        """
        from isaaclab.utils.warp import raycast_mesh

        N = patches.shape[0]
        if N == 0:
            return patches

        # Raise rays 0.5 m above the floor to clear low obstacles (crates, etc.)
        # while still hitting full-height walls.
        lateral_z_offset = 0.5

        # 4 cardinal horizontal directions: +x, -x, +y, -y
        card_dirs = torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
            device=self.device,
        )  # (4, 3)

        # Build (N*4, 3) ray starts and directions
        starts = patches.unsqueeze(1).expand(-1, 4, -1).clone().reshape(-1, 3)
        starts[:, 2] = starts[:, 2] + lateral_z_offset
        dirs = card_dirs.unsqueeze(0).expand(N, -1, -1).reshape(-1, 3)

        # Note: We do NOT use return_distance=True because of IsaacLab bug (see file
        # header note). raycast_mesh tries view(shape[0], shape[1]) on 2D inputs
        # which crashes. Compute distance manually from ray_hits instead.
        ray_hits, _, _, _ = raycast_mesh(starts, dirs, wp_mesh)
        # ray_hits: (N*4, 3) — inf components for rays that miss all geometry

        # Compute horizontal distance from start to hit (rays are purely horizontal)
        # For missed rays (inf), this gives inf distance which is correctly rejected.
        hit_distances = torch.norm(ray_hits[..., :2] - starts[..., :2], dim=-1)  # (N*4,)
        hit_distances = hit_distances.reshape(N, 4)  # (N, 4)

        # Keep only patches where ALL 4 rays hit something within max_wall_distance.
        # inf > max_wall_distance is True, so open-sky rays are caught automatically.
        indoor_mask = (hit_distances <= max_wall_distance).all(dim=1)
        return patches[indoor_mask]

    # ------------------------------------------------------------------
    # CommandTerm overrides
    # ------------------------------------------------------------------

    def _resample_command(self, env_ids: Sequence[int]):
        """Randomly sample a target from the pre-sampled flat patches."""
        num_targets = self.valid_targets.shape[2]
        ids = torch.randint(0, num_targets, size=(len(env_ids),), device=self.device)

        # USD terrain is a single tile → always index at (row=0, col=0)
        self.pos_command_w[env_ids] = self.valid_targets[0, 0, ids]

        # Offset z by the robot's default root height (same as TerrainBasedPose2dCommand)
        self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]

        if self.cfg.simple_heading:
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()

            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            r = torch.empty(len(env_ids), device=self.device)
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)


@configclass
class IndoorPose2dCommandCfg(UniformPose2dCommandCfg):
    """Configuration for :class:`IndoorPose2dCommand`.

    Valid goal positions are found automatically at initialisation time by raycasting
    on the imported USD terrain mesh.  The :attr:`flat_patch_sampling` field mirrors the
    same :class:`~isaaclab.terrains.FlatPatchSamplingCfg` used by the terrain generator
    for generated terrains.

    Only ``ranges.heading`` is required; ``ranges.pos_x`` / ``ranges.pos_y`` from the
    parent class are unused (positions come from flat-patch sampling).
    """

    class_type: type = IndoorPose2dCommand

    flat_patch_sampling: FlatPatchSamplingCfg = MISSING
    """Configuration for sampling obstacle-free floor patches in the indoor environment."""

    indoor_filter_wall_distance: float = 50.0
    """Maximum horizontal distance (m) a cardinal ray may travel before it must hit a wall.

    After flat patches are sampled, 4 horizontal rays (±x, ±y) are cast from each patch.
    A patch is kept only when *every* ray hits geometry within this distance, ensuring the
    patch lies inside an enclosed building rather than in an outdoor open area.

    Increase this value for larger buildings; decrease it to be more conservative.
    A ray returning inf (no hit at all) is always treated as an open direction and the
    patch is rejected regardless of this threshold.
    """

    @configclass
    class Ranges:
        """Only heading is sampled; x/y positions come from flat-patch sampling."""
        heading: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
