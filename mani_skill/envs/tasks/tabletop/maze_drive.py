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


@register_env("MazeDrive-v1", max_episode_steps=300)
class MazeDriveEnv(BaseEnv):
    """
    **Task Description:**
    Navigate a cube through a zigzag obstacle course. Minimize collisions and time.
    
    **Randomizations:**
    - Obstacle positions slightly randomized (±0.05m)
    
    **Rewards:**
    - Dense: -distance_to_next_waypoint + collision_penalty
    - Success: reached goal + bonus for low collision count
    
    **Success Conditions:**
    - Cube passes through all waypoints in order with <5 collisions
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    cube_half_size: float = 0.02
    obstacle_half_size: float = 0.03  # thin walls
    obstacle_stack_levels: int = 6
    goal_radius: float = 0.08
    goal_x: float = 0.18
    obstacle_y_range: float = 0.10
    center_block_half_range: float = 0.02
    obstacle_x_spacing: float = 0.15
    goal_obstacle_clearance_x: float = 0.05
    
    num_slaloms: int = 1  # single obstacle column in front of the goal

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.obstacles = []  # left and right walls
        self.obstacle_stacks = []
        self.obstacle_x_positions = []
        self.wall_block_positions = []
        self.waypoints = []  # target positions through course
        self.obj = None
        self.collision_count = None
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
        pose = sapien_utils.look_at(eye=[0.0, 0.5, 0.6], target=[0.0, 0, 0.15])
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
        pose = sapien_utils.look_at([0.4, 1.0, 0.6], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.4, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.obstacles = []
        self.obstacle_stacks = []
        self.obstacle_x_positions = []
        self.wall_block_positions = []
        self.waypoints = []
        z = self.cube_half_size
        obstacle_base_z = self.obstacle_half_size
        step = 2 * self.obstacle_half_size
        wall_y_positions = np.linspace(-0.12, 0.12, 5)
        wall_z_positions = obstacle_base_z + np.arange(3) * step
        stack = []
        for z_idx, z_pos in enumerate(wall_z_positions):
            for y_idx, y_pos in enumerate(wall_y_positions):
                obs = actors.build_cube(
                    self.scene,
                    half_size=self.obstacle_half_size,
                    color=np.array([0, 0, 0, 1]),
                    name=f"wall_obstacle_{z_idx}_{y_idx}",
                    body_type="dynamic",
                    initial_pose=sapien.Pose(
                        p=[0.065, float(y_pos), float(z_pos)]
                    ),
                )
                self.obstacles.append(obs)
                self.wall_block_positions.append((0.065, float(y_pos), float(z_pos)))
                stack.append(obs)
        self.obstacle_stacks.append(stack)
        self.obstacle_x_positions.append(0.065)
        self.waypoints.append(torch.tensor([0.065, 0.0, z], dtype=torch.float32))

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[self.goal_x, 0, 1e-3]),
        )

        # self.obj = actors.build_cube(
        #     self.scene,
        #     half_size=self.cube_half_size,
        #     color=np.array([1, 0, 0, 1]),
        #     name="cube",
        #     body_type="dynamic",
        #     initial_pose=sapien.Pose(p=[-0.25, 0, self.cube_half_size]),
        # )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # # Reset cube position
            # cube_xyz = torch.tensor(
            #     [-0.25, 0, self.cube_half_size],
            #     device=self.device,
            # ).repeat(b, 1)

            # self.obj.set_pose(
            #     Pose.create_from_pq(p=cube_xyz, q=[1, 0, 0, 0])
            # )

            # Goal at end
            goal_xyz = torch.tensor(
                [self.goal_x, 0, 1e-3],
                device=self.device,
            ).repeat(b, 1)

            self.goal_region.set_pose(
                Pose.create_from_pq(p=goal_xyz, q=euler2quat(0, np.pi / 2, 0))
            )

            obstacle_y = torch.zeros((b, self.num_slaloms), device=self.device)

            for gate_idx, stack in enumerate(self.obstacle_stacks):
                for block_idx, obs in enumerate(stack):
                    block_x, block_y, block_z = self.wall_block_positions[block_idx]
                    p = torch.zeros((b, 3), device=self.device)
                    p[:, 0] = block_x
                    p[:, 1] = block_y
                    p[:, 2] = block_z
                    obs.set_pose(Pose.create_from_pq(p=p, q=[1, 0, 0, 0]))
                    obs.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                    obs.set_angular_velocity(torch.zeros((b, 3), device=self.device))
            
            self.collision_count = torch.zeros(b, device=self.device)

    def evaluate(self):
        # Success: end effector reaches the goal region with few collisions
        tcp_to_goal = self.agent.tcp.pose.p - self.goal_region.pose.p
        is_at_goal = torch.linalg.norm(tcp_to_goal, axis=1) < self.goal_radius

        low_collisions = self.collision_count < 5

        return {
            "success": is_at_goal & low_collisions,
            "reached_goal": is_at_goal,
            "collisions": self.collision_count,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                goal_pos=self.goal_region.pose.p,
                collisions=self.collision_count,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        # Distance from end effector to goal
        dist_to_goal = torch.linalg.norm(
            self.agent.tcp.pose.p - self.goal_region.pose.p, axis=1
        )
        reaching_reward = torch.exp(-3.0 * dist_to_goal)

        # Collision penalty (discourage crashing into obstacles)
        collision_penalty = self.collision_count * 0.2

        # Success bonus
        success_bonus = info["success"].float() * 2.0

        return reaching_reward - collision_penalty + success_bonus

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs, action, info) / 3.0