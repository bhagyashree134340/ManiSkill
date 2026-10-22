"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill tasks can minimally be defined by how the environment resets, what agents/objects are
loaded, goal parameterization, and success conditions

Environment reset is comprised of running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the positions of all objects (called actors), articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do
"""

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
import json
from pathlib import Path


PREFERENCE_FILE = Path("human_preferences.jsonl")



@register_env("WallBypass-v1", max_episode_steps=300)
class WallBypassEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to push and move a cube to a goal region in front of it

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the target goal region is marked by a red/white circular target. The position of the target is fixed to be the cube xy position + [0.1 + goal_radius, 0]

    **Success Conditions:**
    - the cube's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance and the cube is still on the table.
    """

    _sample_video_link = "https://github.com/mani-skill/ManiSkill/raw/main/figures/environment_demos/PushCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ("sparse", "dense", "normalized_dense", "none")

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # set some commonly used values
    wall_length: int = 3  # number of blocks in the wall
    wall_height: int = 4  # number of blocks in the wall
    block_half_size: float = 0.03  # size of each block in the wall

    goal_radius = 0.1
    cube_half_size = 0.02

    # Fixed world-space goal for the current TCP reach task.
    # Match the maze-drive goal spacing for a consistent obstacle layout.
    goal_xy = np.array([0.18, 0.0], dtype=np.float32)

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.wall_blocks = []
        # Store the original pose of every dynamic wall block so that
        # every episode reset restores the walls to their initial state.
        self.initial_wall_positions = []
        self.collision_steps = None
        self.collision_events = None
        self._collision_active = None
        self.cube_displacement = None
        self.previous_wall_positions = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
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
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        # Match the starting lane used by the maze-drive task.
        super()._load_agent(options, sapien.Pose(p=[-0.3, 0, 0]))

    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # ------------------------------------------------------------
        # Maze-drive style wall
        # ------------------------------------------------------------
        # Build a compact obstacle wall matching the maze-drive layout:
        # a vertical stack of short blocks spanning a narrow Y-range in
        # front of the goal, so the robot must navigate around it.
        # ------------------------------------------------------------

        wall_y_positions = np.linspace(-0.12, 0.12, 5)
        wall_z_positions = self.block_half_size + np.arange(3) * (2 * self.block_half_size)

        self.wall_blocks = []
        for z_idx, z_pos in enumerate(wall_z_positions):
            for y_idx, y_pos in enumerate(wall_y_positions):
                block_position = [0.065, float(y_pos), float(z_pos)]
                block = actors.build_cube(
                    scene=self.scene,
                    half_size=self.block_half_size,
                    color=np.array([12, 42, 160, 255]) / 255,
                    name=f"block_{z_idx}_{y_idx}",
                    body_type="dynamic",
                    initial_pose=sapien.Pose(p=block_position),
                )
                self.wall_blocks.append(block)
                self.initial_wall_positions.append(
                    np.asarray(block_position, dtype=np.float32)
                )

        # Optional second wall layout.
        # Uncomment this block if you want a second barrier in the scene.
        for z_idx, z_pos in enumerate(wall_z_positions):
            for y_idx, y_pos in enumerate(wall_y_positions):
                block_position = [float(y_pos), 0.3, float(z_pos)]
                block2 = actors.build_cube(
                    scene=self.scene,
                    half_size=self.block_half_size,
                    color=np.array([12, 42, 160, 255]) / 255,
                    name=f"block2_{z_idx}_{y_idx}",
                    body_type="dynamic",
                    initial_pose=sapien.Pose(p=block_position),
                )
                self.wall_blocks.append(block2)
                self.initial_wall_positions.append(
                    np.asarray(block_position, dtype=np.float32)
                )

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, -1, 1e-3]),
        )

        # optionally you can automatically hide some Actors from view by appending to the self._hidden_objects list. When visual observations
        # are generated or env.render_sensors() is called or env.render() is called with render_mode="sensors", the actor will not show up.
        # This is useful if you intend to add some visual goal sites as e.g. done in PickCube that aren't actually part of the task
        # and are there just for generating evaluation videos.
        # self._hidden_objects.append(self.goal_region)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            self.table_scene.initialize(env_idx)

            # ------------------------------------------------------------
            # RESET ALL DYNAMIC WALL BLOCKS
            # ------------------------------------------------------------
            # The wall blocks are dynamic, so the robot can physically move
            # them.  Resetting only the collision counters is not enough:
            # their pose and velocity must also be restored here.
            #
            # ManiSkill masks set_pose/set_velocity operations using env_idx,
            # so the tensors below can contain all environments while only
            # the environments being reset are modified.

            if len(self.initial_wall_positions) != len(self.wall_blocks):
                raise RuntimeError(
                    "Initial wall state does not match wall_blocks."
                )

            all_wall_positions = torch.tensor(
                np.asarray(self.initial_wall_positions),
                device=self.device,
                dtype=torch.float32,
            )

            identity_quat = torch.tensor(
                [1.0, 0.0, 0.0, 0.0],
                device=self.device,
                dtype=torch.float32,
            )

            for block_idx, block in enumerate(self.wall_blocks):
                position = all_wall_positions[block_idx].unsqueeze(0).repeat(
                    self.num_envs, 1
                )
                quaternion = identity_quat.unsqueeze(0).repeat(
                    self.num_envs, 1
                )

                block.set_pose(
                    Pose.create_from_pq(
                        p=position,
                        q=quaternion,
                    )
                )

                block.set_linear_velocity(
                    torch.zeros(
                        (self.num_envs, 3),
                        device=self.device,
                        dtype=torch.float32,
                    )
                )

                block.set_angular_velocity(
                    torch.zeros(
                        (self.num_envs, 3),
                        device=self.device,
                        dtype=torch.float32,
                    )
                )

            n_envs = len(env_idx)
            goal_xy = torch.tensor(self.goal_xy, device=self.device, dtype=torch.float32)
            target_region_xyz = torch.zeros((n_envs, 3), device=self.device)
            target_region_xyz[..., :2] = goal_xy.tile((n_envs, 1))
            target_region_xyz[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

            # Keep reset side effects minimal; run reachability checks explicitly.

            if self.collision_steps is None or self.collision_steps.shape[0] != self.num_envs:
                self.collision_steps = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.int64
                )
                self.collision_events = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.int64
                )
                self._collision_active = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.bool
                )

            self.collision_steps[env_idx] = 0
            self.collision_events[env_idx] = 0
            self._collision_active[env_idx] = False
            self.previous_wall_positions = torch.stack([block.pose.p.clone() for block in self.wall_blocks], dim=1)

    def _robot_wall_collision(self) -> torch.Tensor:
        collision = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for link in self.agent.robot.get_links():
            for wall_block in self.wall_blocks:
                contact_forces = self.scene.get_pairwise_contact_forces(link, wall_block)
                collision = collision | (
                    torch.linalg.norm(contact_forces, dim=-1) > 1e-4
                )
        return collision

    def step(self, action: Union[None, Array, dict[str, Union[np.ndarray, torch.Tensor]]]):
        action = self._step_action(action)
        self._elapsed_steps += 1

        collision = self._robot_wall_collision()
        if self.collision_steps is None or self.collision_events is None or self._collision_active is None:
            raise RuntimeError("Collision counters were not initialized before step().")

        self.collision_steps = self.collision_steps + collision.to(torch.int64)
        new_collision = collision & (~self._collision_active)
        self.collision_events = self.collision_events + new_collision.to(torch.int64)
        self._collision_active = collision

        info = self.get_info()
        obs = self.get_obs(info, unflattened=True)
        
        # print("WALL POSITIONS:")
        # print(obs["extra"]["wall_positions"])

        reward = self.get_reward(obs=obs, action=action, info=info)
        obs = self._flatten_raw_obs(obs)
        if "success" in info:
            if "fail" in info:
                terminated = torch.logical_or(info["success"], info["fail"])
            else:
                terminated = info["success"].clone()
        else:
            if "fail" in info:
                terminated = info["fail"].clone()
            else:
                terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_obs = obs
        return (
            obs,
            reward,
            terminated,
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            info,
        )

    def get_info(self) -> dict:
        info = super().get_info()
        if self.collision_steps is None or self.collision_events is None or self._collision_active is None:
            info.update(
                collision=torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
                collision_steps=torch.zeros(self.num_envs, device=self.device, dtype=torch.int64),
                collision_events=torch.zeros(self.num_envs, device=self.device, dtype=torch.int64),
            )
            return info

        info.update(
            collision=self._collision_active.clone(),
            collision_steps=self.collision_steps.clone(),
            collision_events=self.collision_events.clone(),
        )
        return info

    def evaluate(self):
        # Active task: end effector reaches the target region in 3D while the wall remains.
        tcp_to_goal_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - self.goal_region.pose.p, axis=1
        )
        is_tcp_at_goal = tcp_to_goal_dist < self.goal_radius

        # Legacy cube-at-goal logic retained here for easy reactivation later:
        # cube_to_goal_dist = torch.linalg.norm(
        #     self.obj.pose.p - self.goal_region.pose.p, axis=1
        # )
        # is_cube_at_goal = (
        #     cube_to_goal_dist < self.goal_radius
        # ) & (self.obj.pose.p[..., 2] < self.cube_half_size + 5e-3)

        return {
            "success": is_tcp_at_goal,
        }

    def _get_obs_extra(self, info: dict):

        wall_positions = torch.stack([block.pose.p for block in self.wall_blocks], dim=1).flatten(start_dim=1)
        # breakpoint()

        # Active task: achieved_goal is the TCP position, desired_goal is the target.
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            achieved_goal=self.agent.tcp.pose.p,
            desired_goal=self.goal_region.pose.p,
            wall_positions=wall_positions,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                goal_pos=self.goal_region.pose.p,
                # obj_pose=self.obj.pose.raw_pose,
            )
        return obs

        # Legacy cube-at-goal observation logic retained below for later reuse:
        # obs = dict(
        #     tcp_pose=self.agent.tcp.pose.raw_pose,
        #     achieved_goal=self.obj.pose.p[..., :2],
        #     desired_goal=self.goal_region.pose.p[..., :2],
        # )
        # if self.obs_mode_struct.use_state:
        #     obs.update(
        #         goal_pos=self.goal_region.pose.p,
        #         obj_pose=self.obj.pose.raw_pose,
        #     )
        # return obs

        # New fixed logic for the TCP reach task (kept after the original code):
        # obs = dict(
        #     tcp_pose=self.agent.tcp.pose.raw_pose,
        #     achieved_goal=self.agent.tcp.pose.p[..., :2],
        #     desired_goal=self.goal_region.pose.p[..., :2],
        # )
        # if self.obs_mode_struct.use_state:
        #     obs.update(goal_pos=self.goal_region.pose.p)
        # return obs

    def compute_sparse_reward(self, obs: Any, action: Array, info: dict):
        if "success" in info:
            return info["success"].to(torch.float)
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        return info["success"].float()


    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_her_reward(
            self.agent.tcp.pose.p,
            self.goal_region.pose.p,
        )

    def compute_her_reward(self, achieved_goal, desired_goal):
        distance = torch.linalg.norm(
            achieved_goal - desired_goal,
            dim=-1
        )

        return (distance < self.goal_radius).float()

    def check_waypoint_path(self):
        """
        Test whether the Panda can reach the goal by moving through the
        opening on +Y and checking collisions against every wall block.
        """

        waypoint_positions = torch.tensor(
            [
                [0.00, 0.30, 0.20],
                [0.20, 0.30, 0.20],
                [0.20, 0.10, 0.20],
                [0.20, 0.00, 0.10],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        arm_controller = self.agent.controller.controllers["arm"]
        arm_joint_indices = arm_controller.active_joint_indices
        current_qpos = self.agent.robot.get_qpos().clone()
        current_qvel = self.agent.robot.get_qvel().clone()
        if current_qpos.ndim == 1:
            current_qpos = current_qpos.unsqueeze(0)
        if current_qvel.ndim == 1:
            current_qvel = current_qvel.unsqueeze(0)

        print("\n========== WAYPOINT PATH TEST ==========")

        qpos_waypoints = []
        q0 = current_qpos

        for i, position in enumerate(waypoint_positions):
            target_pose = Pose.create_from_pq(
                p=position.unsqueeze(0),
                q=euler2quat(0, np.pi / 2, 0),
            )

            ik_result = arm_controller.kinematics.compute_ik(
                pose=target_pose,
                q0=q0,
            )

            print(f"\nWaypoint {i}: {position.cpu().numpy()}")

            if ik_result is None:
                print("  IK: FAILED")
                self.agent.robot.set_qpos(current_qpos)
                self.agent.robot.set_qvel(current_qvel)
                print("\n===================================\n")
                return False

            if ik_result.ndim == 1:
                ik_result = ik_result.unsqueeze(0)

            print("  IK: SUCCESS")
            print("  Joint configuration:")
            print(ik_result[0].detach().cpu().numpy())
            waypoint_qpos = q0.clone()
            waypoint_qpos[:, arm_joint_indices] = ik_result
            qpos_waypoints.append(waypoint_qpos.clone())
            q0 = waypoint_qpos

        path_steps = 50 #12

        def check_segment(start_qpos, end_qpos):
            for step_idx, alpha in enumerate(
                torch.linspace(0.0, 1.0, path_steps, device=self.device)
            ):
                waypoint_qpos = start_qpos + alpha * (end_qpos - start_qpos)
                self.agent.robot.set_qpos(waypoint_qpos)
                self.agent.robot.set_qvel(torch.zeros_like(current_qvel))
                self.scene.step()

                for link in self.agent.robot.get_links():
                    for wall_block in self.wall_blocks:
                        contact_forces = self.scene.get_pairwise_contact_forces(
                            link, wall_block
                        )
                        if torch.any(torch.linalg.norm(contact_forces, dim=-1) > 1e-4):
                            return False, step_idx

            return True, None

        start_qpos = current_qpos
        segments = [
            (start_qpos, qpos_waypoints[0]),
            (qpos_waypoints[0], qpos_waypoints[1]),
            (qpos_waypoints[1], qpos_waypoints[2]),
            (qpos_waypoints[2], qpos_waypoints[3]),
        ]

        path_clear = True
        blocked_segment = None
        blocked_step = None

        for segment_idx, (start, end) in enumerate(segments):
            segment_clear, collision_step = check_segment(start, end)
            if not segment_clear:
                path_clear = False
                blocked_segment = segment_idx
                blocked_step = collision_step
                break

        self.agent.robot.set_qpos(current_qpos)
        self.agent.robot.set_qvel(current_qvel)

        if path_clear:
            print("\nPATH THROUGH OPENING: CLEAR\n")
            return True

        print(f"\nPATH THROUGH OPENING: BLOCKED (segment {blocked_segment}, step {blocked_step})\n")
        return False

    # Legacy cube-at-goal dense reward logic retained for later reuse:
    # def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
    #     return self.compute_her_reward(
    #         self.obj.pose.p[..., :2],
    #         self.goal_region.pose.p[..., :2],
    #     )
    #
    # def compute_her_reward(self, achieved_goal, desired_goal):
    #     distance = torch.linalg.norm(achieved_goal - desired_goal, dim=-1)
    #     return (distance < self.goal_radius).float()


    def wall_displacement(self):
        wall_positions = torch.stack([block.pose.p for block in self.wall_blocks], dim=1)
        displacement = torch.linalg.norm(wall_positions - self.previous_wall_positions, dim=-1).sum(dim=1)
        self.previous_wall_positions = wall_positions.clone()
        self.cube_displacement = displacement
        return displacement