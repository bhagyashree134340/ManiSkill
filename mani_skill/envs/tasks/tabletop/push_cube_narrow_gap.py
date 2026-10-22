from typing import Any, Union

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


@register_env("PushThroughNarrowGap-v1", max_episode_steps=250)
class PushThroughNarrowGapEnv(BaseEnv):
    """
    **Task Description:**
    Push a cube through a narrow gap without colliding. Balance speed vs precision.
    
    **Challenge:**
    - Gap is only slightly wider than cube (0.05m cube, 0.06m gap)
    - Too fast → collision with walls
    - Too slow → episode timeout
    
    **Rewards:**
    - Success: cube fully passes through gap
    - Dense: speed bonus - collision penalty
    
    **Success Conditions:**
    - Cube x position > 0.1 (past the gap) with minimal collisions
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    cube_half_size: float = 0.025
    gap_width: float = 0.06  # only 0.01m clearance on each side
    wall_thickness: float = 0.04
    goal_x: float = 0.15  # how far past gap = success

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.left_wall = None
        self.right_wall = None
        self.obj = None
        self.collision_count = None
        self.collision_since_last_step = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0.3, 0.5], target=[0.05, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, 0.5, 0.5], [0.05, 0, 0.1])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Build narrow gap walls
        gap_center_y = 0.0
        
        # Left wall (kinematic, so it doesn't move)
        self.left_wall = actors.build_cube(
            self.scene,
            half_size=[self.wall_thickness, (self.gap_width / 2) + 0.01, 0.1],
            color=np.array([0.5, 0.5, 0.5, 1]),
            name="left_wall",
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0.05, -(self.gap_width / 2 + 0.01), 0.1]),
        )

        # Right wall
        self.right_wall = actors.build_cube(
            self.scene,
            half_size=[self.wall_thickness, (self.gap_width / 2) + 0.01, 0.1],
            color=np.array([0.5, 0.5, 0.5, 1]),
            name="right_wall",
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0.05, (self.gap_width / 2 + 0.01), 0.1]),
        )

        # Cube to push through
        self.obj = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([1, 0, 0, 1]),
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[-0.1, 0, self.cube_half_size]),
        )

        # Goal marker (invisible, just for reference)
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=0.05,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[self.goal_x, 0, 1e-3]),
        )
        self._hidden_objects.append(self.goal_region)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Reset cube to starting position (before gap)
            cube_xyz = torch.tensor(
                [-0.1, 0, self.cube_half_size],
                device=self.device,
            ).repeat(b, 1)

            self.obj.set_pose(
                Pose.create_from_pq(p=cube_xyz, q=[1, 0, 0, 0])
            )

            self.collision_count = torch.zeros(b, device=self.device)
            self.collision_since_last_step = torch.zeros(b, device=self.device)

    def _detect_collision_this_step(self):
        """Check if cube collided with walls this step"""
        # Simple heuristic: if cube's y position is outside safe zone
        cube_y = self.obj.pose.p[..., 1]
        in_danger_zone = torch.abs(cube_y) > (self.gap_width / 2)
        return in_danger_zone

    def evaluate(self):
        # Success: cube past the gap with few collisions
        is_past_gap = self.obj.pose.p[..., 0] > self.goal_x
        low_collisions = self.collision_count < 3

        return {
            "success": is_past_gap & low_collisions,
            "past_goal": is_past_gap,
            "collisions": self.collision_count,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                obj_y=self.obj.pose.p[..., 1],  # proximity to walls
                gap_width=torch.tensor(self.gap_width, device=self.device),
                collisions=self.collision_count,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        # Progress reward (distance towards goal)
        progress = torch.clamp(self.obj.pose.p[..., 0] / self.goal_x, 0, 1)
        progress_reward = progress * 0.5
        
        # Speed bonus (reach goal faster, but not recklessly)
        speed_bonus = 0.01  # small per-step bonus for being alive
        
        # Collision penalty (getting worse each collision)
        collision_penalty = self.collision_count * 0.5
        
        # Success bonus
        success_bonus = info["success"].float() * 2.0
        
        return progress_reward + speed_bonus - collision_penalty + success_bonus

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs, action, info) / 3.0