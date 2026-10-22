import argparse
import signal

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, visualization
from mani_skill.utils.wrappers import RecordEpisode

import pyspacemouse

# with pyspacemouse.open(device_index=1) as device:
#     state = device.read()

device = pyspacemouse.open()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, required=True)
    parser.add_argument("-o", "--obs-mode", type=str)
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-c", "--control-mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument("--render-mode", type=str, default="sensors")
    parser.add_argument("--enable-sapien-viewer", action="store_true")
    parser.add_argument("--record-dir", type=str)
    args, opts = parser.parse_known_args()

    # Parse env kwargs
    print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs

    return args


def main():
    
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()

    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        **args.env_kwargs
    )

    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir)

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Control mode", env.unwrapped.control_mode)
    print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset()
    after_reset = True

    # Viewer
    if args.enable_sapien_viewer:
        env.unwrapped.render_human()

    renderer = visualization.ImageRenderer()
    # disable all default plt shortcuts that are lowercase letters
    plt.rcParams["keymap.fullscreen"].remove("f")
    plt.rcParams["keymap.home"].remove("h")
    plt.rcParams["keymap.home"].remove("r")
    plt.rcParams["keymap.back"].remove("c")
    plt.rcParams["keymap.forward"].remove("v")
    plt.rcParams["keymap.pan"].remove("p")
    plt.rcParams["keymap.zoom"].remove("o")
    plt.rcParams["keymap.save"].remove("s")
    plt.rcParams["keymap.grid"].remove("g")
    plt.rcParams["keymap.yscale"].remove("l")
    plt.rcParams["keymap.xscale"].remove("k")

    def render_wait():
        if not args.enable_sapien_viewer:
            return
        while True:
            env.unwrapped.render_human()  
            sapien_viewer = env.unwrapped.viewer
            if sapien_viewer.window.key_down("0"):
                break

    # Embodiment
    has_base = "base" in env.unwrapped.agent.controller.configs
    num_arms = sum("arm" in x for x in env.unwrapped.agent.controller.configs)
    has_gripper = any("gripper" in x for x in env.unwrapped.agent.controller.configs)
    gripper_action = 1
    EE_ACTION = 0.1
    

    while True:

        state = device.read()
        # -------------------------------------------------------------------------- #
        # Visualization
        # -------------------------------------------------------------------------- #
        if args.enable_sapien_viewer:
            env.unwrapped.render_human()


        rendered = env.render()
        if hasattr(rendered, "cpu"):
            render_frame = rendered.cpu().numpy()[0]
        elif isinstance(rendered, np.ndarray):
            render_frame = rendered[0] if rendered.ndim == 4 else rendered
        else:
            render_frame = None

        if after_reset:
            after_reset = False
            # Re-focus on opencv viewer
            if args.enable_sapien_viewer:
                renderer.close()
                renderer = visualization.ImageRenderer()
                pass
        # -------------------------------------------------------------------------- #
        # Interaction
        # -------------------------------------------------------------------------- #
        # Input
        if render_frame is not None:
            renderer(render_frame)

        # if render_frame is not None:
        #     plt.figure()
        #     plt.imshow(render_frame)
        #     plt.show(block=True)

        # print("rendered:", type(rendered), getattr(rendered, "shape", None))
        # print("render_frame:", type(render_frame), getattr(render_frame, "shape", None))

        # if render_frame is not None:
        #     print(
        #         "pixels:",
        #         "min =", render_frame.min(),
        #         "max =", render_frame.max(),
        #         "mean =", render_frame.mean()
        #     )

        # key = opencv_viewer.imshow(render_frame.cpu().numpy()[0])

        key = renderer.last_event.key if renderer.last_event is not None else None
        body_action = np.zeros([3])
        base_action = np.zeros([2])  # hardcoded for fetch robot

        state = device.read()

        # Parse end-effector action
        if (
            "pd_ee_delta_pose" in args.control_mode
            or "pd_ee_target_delta_pose" in args.control_mode
        ):
            ee_action = np.zeros([6])
        elif (
            "pd_ee_delta_pos" in args.control_mode
            or "pd_ee_target_delta_pos" in args.control_mode
        ):
            ee_action = np.zeros([3])
        else:
            raise NotImplementedError(args.control_mode)

        # Base. Hardcoded for Fetch robot at the moment. In the future write interface to do this
        if has_base:
            if key == "w":  # forward
                base_action[0] = 1
            elif key == "s":  # backward
                base_action[0] = -1
            elif key == "a":  # rotate counter
                base_action[1] = 1
            elif key == "d":  # rotate clockwise
                base_action[1] = -1
            elif key == "z":  # lift
                body_action[2] = 1
            elif key == "x":  # lower
                body_action[2] = -1
            elif key == "v":  # rotate head left
                body_action[0] = 1
            elif key == "b":  # rotate head right
                body_action[0] = -1
            elif key == "n":  # tilt head down
                body_action[1] = 1
            elif key == "m":  # rotate head up
                body_action[1] = -1

        # End-effector
        if num_arms > 0:
            # Position

            
            

            ee_action[0] = state.x
            ee_action[1] = state.y
            ee_action[2] = state.z
            ee_action[3] = state.roll
            ee_action[4] = state.pitch
            ee_action[5] = state.yaw

            print(ee_action)


            # if key == "i":  # +x
            #     ee_action[0] = EE_ACTION
            # elif key == "k":  # -x
            #     ee_action[0] = -EE_ACTION
            # elif key == "j":  # +y
            #     ee_action[1] = EE_ACTION
            # elif key == "l":  # -y
            #     ee_action[1] = -EE_ACTION
            # elif key == "u":  # +z
            #     ee_action[2] = EE_ACTION
            # elif key == "o":  # -z
            #     ee_action[2] = -EE_ACTION

            # # Rotation (axis-angle)
            # if key == "1":
            #     ee_action[3:6] = (1, 0, 0)
            # elif key == "2":
            #     ee_action[3:6] = (-1, 0, 0)
            # elif key == "3":
            #     ee_action[3:6] = (0, 1, 0)
            # elif key == "4":
            #     ee_action[3:6] = (0, -1, 0)
            # elif key == "5":
            #     ee_action[3:6] = (0, 0, 1)
            # elif key == "6":
            #     ee_action[3:6] = (0, 0, -1)

        # Gripper
        if has_gripper:
            if key == "f":  # open gripper
                gripper_action = 1
            elif key == "g":  # close gripper
                gripper_action = -1

        # Other functions
        if key == "0":  # switch to SAPIEN viewer
            render_wait()
        elif key == "r":  # reset env
            obs, _ = env.reset()
            gripper_action = 1
            after_reset = True
            continue
        elif key == "q":  # exit
            break

        # Visualize observation
        if key == "v":
            if "pointcloud" in env.unwrapped.obs_mode:
                import trimesh

                xyzw = obs["pointcloud"]["xyzw"]
                mask = xyzw[..., 3] > 0
                rgb = obs["pointcloud"]["rgb"]
                if "robot_seg" in obs["pointcloud"]:
                    robot_seg = obs["pointcloud"]["robot_seg"]
                    rgb = np.uint8(robot_seg * [11, 61, 127])
                trimesh.PointCloud(xyzw[mask, :3], rgb[mask]).show()

        # -------------------------------------------------------------------------- #
        # Post-process action
        # -------------------------------------------------------------------------- #
        action_dict = dict(
            base=base_action, arm=ee_action, body=body_action, gripper=gripper_action
        )
        action_dict = common.to_tensor(action_dict)
        action = env.unwrapped.agent.controller.from_action_dict(action_dict)

        obs, reward, terminated, truncated, info = env.step(action)
        print("reward", reward)
        print("terminated", terminated, "truncated", truncated)
        print("info", info)

    env.close()


if __name__ == "__main__":
    main()
