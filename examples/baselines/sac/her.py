
import torch

from third_party.ManiSkill.examples.baselines.sac.replay_buffer import ReplayBufferSample


class HERReplayBufferSample:
    def __init__(self, obs, next_obs, actions, rewards, dones, costs):
        # Names used by your modified SAC-HER
        self.obs = obs
        self.next_obs = next_obs

        # Names expected by SB3 / CSAC-LB
        self.observations = obs
        self.next_observations = next_obs
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.costs = costs




def infer_goal_slice(env):
    """Infer the flattened desired_goal slice from a raw Gymnasium env."""
    base_env = env
    while hasattr(base_env, "env") and not hasattr(base_env, "_get_obs_state_dict"):
        base_env = base_env.env

    if not hasattr(base_env, "_get_obs_state_dict"):
        raise ValueError(
            "Cannot infer desired_goal slice: environment does not expose "
            "_get_obs_state_dict(). Pass goal_slice explicitly."
        )

    state = base_env._get_obs_state_dict({})
    offset = 0
    goal_start = None
    goal_end = None

    def walk(value):
        nonlocal offset, goal_start, goal_end
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "desired_goal":
                    arr = torch.as_tensor(item)
                    goal_start = offset
                    goal_end = offset + arr.numel()
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        else:
            try:
                offset += torch.as_tensor(value).numel()
            except Exception:
                pass

    walk(state)
    if goal_start is None:
        raise ValueError("Could not find desired_goal in the environment observation.")

    # desired_goal is a 3D Cartesian position in WallBypass-v1.
    if goal_end is None:
        goal_end = goal_start + 3

    return slice(goal_start, goal_end)


class HERReplayBuffer:
    def __init__(
        self,
        env=None,
        num_envs: int = 1,
        buffer_size: int = 1_000_000,
        storage_device: torch.device = torch.device("cpu"),
        sample_device: torch.device = torch.device("cpu"),
        her_ratio: float = 0.8,
        goal_radius: float = 0.1,
        device=torch.device("cpu"),
        goal_env=None,
        observation_space=None,
        action_space=None,
        n_envs=None,
        goal_slice=None,
        max_episode_steps=300,
        **kwargs,
    ):
        if n_envs is not None:
            num_envs = n_envs
        self.env = env
        self.goal_env = goal_env if goal_env is not None else env
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs
        self.her_ratio = her_ratio
        self.goal_radius = goal_radius
        self.max_episode_steps = int(getattr(env, "max_episode_steps", max_episode_steps)) if env is not None else int(max_episode_steps)
        self.device = device

        if observation_space is None:
            if env is None:
                raise ValueError("observation_space is required when env is None")
            observation_space = env.single_observation_space
        if action_space is None:
            if env is None:
                raise ValueError("action_space is required when env is None")
            action_space = env.single_action_space

        self.observation_space = observation_space
        self.action_space = action_space

        # SB3 does not pass the environment to replay-buffer construction.
        # Therefore the exact desired_goal slice is supplied by CSAC-LB.
        self.goal_slice = goal_slice if goal_slice is not None else self._infer_goal_slice(self.goal_env)

        # WallBypass-v1 uses a 3D desired goal (x, y, z).
        # Some inference paths return an open-ended slice such as slice(start, None).
        # Convert that to an explicit slice before allocating goal buffers.
        if self.goal_slice.stop is None:
            self.goal_slice = slice(
                self.goal_slice.start,
                self.goal_slice.start + 3,
            )

        # Fast lookup of future transitions by (env_idx, episode_id, step).
        # This avoids the O(buffer_size * batch_size) full-buffer comparison that
        # made HER sampling extremely slow in large replay buffers.
        self.episode_lookup = [dict() for _ in range(self.num_envs)]

        self.obs = torch.zeros(
            (self.per_env_buffer_size, self.num_envs)
            + observation_space.shape,
            device=storage_device,
        )
        self.next_obs = torch.zeros(
            (self.per_env_buffer_size, self.num_envs)
            + observation_space.shape,
            device=storage_device,
        )
        self.actions = torch.zeros(
            (self.per_env_buffer_size, self.num_envs)
            + action_space.shape,
            device=storage_device,
        )
        self.rewards = torch.zeros(
            (self.per_env_buffer_size, self.num_envs),
            device=storage_device,
        )
        self.costs = torch.zeros(
            (self.per_env_buffer_size, self.num_envs),
            device=storage_device,
        )
        self.dones = torch.zeros(
            (self.per_env_buffer_size, self.num_envs),
            device=storage_device,
        )

        goal_dim = self.goal_slice.stop - self.goal_slice.start
        self.goal_dim = int(goal_dim)

        self.achieved_goals = torch.zeros(
            (self.per_env_buffer_size, self.num_envs, self.goal_dim),
            device=storage_device,
        )
        self.desired_goals = torch.zeros(
            (self.per_env_buffer_size, self.num_envs, self.goal_dim),
            device=storage_device,
        )
        self.next_achieved_goals = torch.zeros(
            (self.per_env_buffer_size, self.num_envs, self.goal_dim),
            device=storage_device,
        )

        # Episode ID and timestep for every environment separately
        self.episode_ids = torch.zeros(
            (self.per_env_buffer_size, self.num_envs),
            dtype=torch.long,
            device=storage_device,
        )
        self.episode_steps = torch.zeros(
            (self.per_env_buffer_size, self.num_envs),
            dtype=torch.long,
            device=storage_device,
        )

        self.current_episode_id = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=storage_device,
        )
        self.current_episode_step = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=storage_device,
        )

    def _infer_goal_slice(self, env):
        """Return the flattened desired_goal slice used by WallBypass-v1.

        WallBypass-v1 builds its flattened observation as:

            state observation (18 dims)
            + tcp_pose (7)
            + achieved_goal (3)
            + desired_goal (3)   
            + wall_positions (90)
            + goal_pos (3)

        The previous implementation walked the state dictionary and treated
        its flattened offset as the global observation offset. That is wrong:
        desired_goal lives in the *extra* observation, after the base state.
        """
        base_env = env
        while hasattr(base_env, "env") and not hasattr(base_env, "_get_obs_state_dict"):
            base_env = base_env.env

        if not hasattr(base_env, "_get_obs_state_dict"):
            raise ValueError(
                "Cannot infer desired_goal slice: environment does not expose "
                "_get_obs_state_dict(). Pass goal_slice explicitly."
            )

        # Flattened size of the base/state observation.
        state = base_env._get_obs_state_dict({})

        def numel(value):
            if isinstance(value, dict):
                return sum(numel(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return sum(numel(v) for v in value)
            try:
                return int(torch.as_tensor(value).numel())
            except Exception:
                return 0

        state_dim = numel(state)

        # In WallBypass-v1, _get_obs_extra() is ordered as:
        # tcp_pose (7), achieved_goal (3), desired_goal (3), ...
        # Therefore desired_goal starts 10 entries after the state prefix.
        goal_start = state_dim + 7 + 3
        goal_end = goal_start + 3

        return slice(goal_start, goal_end)

    def compute_reward(
        self,
        achieved_goal: torch.Tensor,
        desired_goal: torch.Tensor,
    ) -> torch.Tensor:
        distance = torch.linalg.norm(
            achieved_goal - desired_goal,
            dim=-1,
        )
        return (distance < self.goal_radius).float()

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        achieved_goal: torch.Tensor,
        desired_goal: torch.Tensor,
        next_achieved_goal: torch.Tensor,
        episode_done: torch.Tensor | None = None,
        cost: torch.Tensor | None = None,
    ):
        if episode_done is None:
            episode_done = done
        if cost is None:
            cost = torch.zeros_like(done, dtype=torch.float32)

        obs = torch.as_tensor(obs)
        next_obs = torch.as_tensor(next_obs)
        action = torch.as_tensor(action)
        reward = torch.as_tensor(reward)
        done = torch.as_tensor(done)
        episode_done = torch.as_tensor(episode_done)
        achieved_goal = torch.as_tensor(achieved_goal)
        desired_goal = torch.as_tensor(desired_goal)
        next_achieved_goal = torch.as_tensor(next_achieved_goal)
        cost = torch.as_tensor(cost, dtype=torch.float32)

        if self.storage_device == torch.device("cpu"):
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()
            episode_done = episode_done.cpu()
            achieved_goal = achieved_goal.cpu()
            desired_goal = desired_goal.cpu()
            next_achieved_goal = next_achieved_goal.cpu()
            cost = cost.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.costs[self.pos] = cost

        self.achieved_goals[self.pos] = achieved_goal
        self.desired_goals[self.pos] = desired_goal
        self.next_achieved_goals[self.pos] = next_achieved_goal

        self.episode_ids[self.pos] = self.current_episode_id
        self.episode_steps[self.pos] = self.current_episode_step

        # Track the exact buffer index for each (env, episode, step) so HER can
        # pick a future goal in O(1) dictionary lookup instead of scanning the
        # entire replay buffer.
        for env_idx in range(self.num_envs):
            episode_id = int(self.current_episode_id[env_idx].item())
            step = int(self.current_episode_step[env_idx].item())
            env_lookup = self.episode_lookup[env_idx]
            env_lookup.setdefault(episode_id, {})[step] = self.pos

        # Advance episode timestep
        self.current_episode_step += 1

        # Reset episode information for environments that actually finished an episode.
        # This must use the real terminal mask, not the bootstrap mask used purely for
        # TD target handling.
        done_mask = episode_done.bool()

        self.current_episode_step[done_mask] = 0
        self.current_episode_id[done_mask] += 1

        self.pos += 1

        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env=None):
        if self.full:
            batch_inds = torch.randint(
                0,
                self.per_env_buffer_size,
                (batch_size,),
                device=self.storage_device,
            )
        else:
            batch_inds = torch.randint(
                0,
                self.pos,
                (batch_size,),
                device=self.storage_device,
            )

        env_inds = torch.randint(
            0,
            self.num_envs,
            (batch_size,),
            device=self.storage_device,
        )

        # Regular transitions
        obs = self.obs[batch_inds, env_inds].to(self.sample_device)
        next_obs = self.next_obs[batch_inds, env_inds].to(self.sample_device)
        actions = self.actions[batch_inds, env_inds].to(self.sample_device)
        rewards = self.rewards[batch_inds, env_inds].to(self.sample_device)
        dones = self.dones[batch_inds, env_inds].to(self.sample_device)
        costs = self.costs[batch_inds, env_inds].to(self.sample_device).reshape(-1, 1)

        # Original desired goals
        desired_goals = self.desired_goals[
            batch_inds, env_inds
        ].to(self.sample_device)

        her_weights = self.her_weight(
            costs.squeeze(-1).to(self.storage_device)
        )

        # Which samples should use HER?
        use_her = (
            torch.rand(
                batch_size,
                device=self.storage_device,
            )
            < self.her_ratio 
        )
        
        # use_her = (
        #     torch.rand(
        #         batch_size,
        #         device=self.storage_device,
        #     )
        #     < self.her_ratio * self.her_weight(costs)
        # )

        if use_her.any():
            her_indices = torch.where(use_her)[0]

            current_steps = self.episode_steps[
                batch_inds[her_indices],
                env_inds[her_indices],
            ]
            target_episode_ids = self.episode_ids[
                batch_inds[her_indices],
                env_inds[her_indices],
            ]

            # Pick a random future offset in the same episode.
            future_offset = torch.randint(
                1,
                self.max_episode_steps,
                (len(her_indices),),
                device=self.storage_device,
            )
            future_steps = torch.clamp(
                current_steps + future_offset,
                max=self.max_episode_steps - 1,
            )

            valid_her_indices = []
            valid_env_indices = []
            valid_future_indices = []

            for i, her_idx in enumerate(her_indices.tolist()):
                env_idx = int(env_inds[her_idx].item())
                episode_id = int(target_episode_ids[i].item())
                future_step = int(future_steps[i].item())
                env_lookup = self.episode_lookup[env_idx]
                episode_lookup = env_lookup.get(episode_id)
                if episode_lookup is None:
                    continue
                future_index = episode_lookup.get(future_step)
                if future_index is None:
                    continue

                valid_her_indices.append(her_idx)
                valid_env_indices.append(env_idx)
                valid_future_indices.append(future_index)

            if valid_her_indices:
                valid_her_tensor = torch.tensor(valid_her_indices, device=self.storage_device)
                valid_env_tensor = torch.tensor(valid_env_indices, device=self.storage_device)
                valid_future_tensor = torch.tensor(valid_future_indices, device=self.storage_device)

                desired_goals[valid_her_tensor] = (
                    self.next_achieved_goals[
                        valid_future_tensor,
                        valid_env_tensor,
                    ].to(self.sample_device)
                )

        # Replace the goal in the observation at the exact flattened index used by the
        # underlying environment. Do not assume the goal lives in the last 2 entries of the
        # vector because WallBypass appends goal_pos/obj_pose after desired_goal.
        obs = obs.clone()
        next_obs = next_obs.clone()

        goal_start = self.goal_slice.start
        goal_end = self.goal_slice.stop

        # SB3/SAC observations can have an extra singleton environment
        # dimension: [batch, 1, obs_dim].
        # desired_goals are sampled as [batch, goal_dim].
        # Expand them to [batch, 1, goal_dim] before inserting them.
        if obs.ndim == 3 and desired_goals.ndim == 2:
            desired_goals_for_obs = desired_goals.unsqueeze(1)
        else:
            desired_goals_for_obs = desired_goals

        obs[..., goal_start:goal_end] = desired_goals_for_obs
        next_obs[..., goal_start:goal_end] = desired_goals_for_obs

        # Compute rewards using the new HER goals
        next_achieved = self.next_achieved_goals[
            batch_inds,
            env_inds,
        ].to(self.sample_device)

        her_rewards = self.compute_reward(
            next_achieved,
            desired_goals,
        )

        rewards[use_her.to(self.sample_device)] = her_rewards[
            use_her.to(self.sample_device)
        ]

        return HERReplayBufferSample(
            obs=obs,
            next_obs=next_obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            costs=costs,
        )

    def her_weight(self, cost):
        return torch.clamp(1.0 - cost, min=0.0)