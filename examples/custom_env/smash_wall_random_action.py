### Make sure to restart the notebook if you already ran a CPU sim!! ###
# Import required packages
import gymnasium as gym
import mani_skill.envs
import torch
import time

num_envs = 2  # you can go up to 4096 on better GPUs
render_mode = (
    "human"  # set to "human" to visualize the environment, or "rgb_array" to get images
)
parallel_in_single_scene = (
    True  # set to False to run each environment in a separate scene
)
env = gym.make(
    "SmashWall-v1",
    num_envs=num_envs,
    render_mode=render_mode,
    parallel_in_single_scene=parallel_in_single_scene,
)
env.unwrapped.print_sim_details()
obs, _ = env.reset(seed=0)
done = False
start_time = time.time()
total_rew = 0
n_steps = 2000
for _ in range(n_steps):
    # note that env.action_space is now a batched action space
    obs, rew, terminated, truncated, info = env.step(
        torch.from_numpy(env.action_space.sample())
    )
    env.render()
    done = (
        terminated | truncated
    ).any()  # stop if any environment terminates/truncates
    if done:
        obs, _ = env.reset()

N = num_envs * n_steps  # total number of frames simulated
dt = time.time() - start_time
FPS = N / (dt)
print(f"Frames Per Second = {N} / {dt} = {FPS}")
