import os
import numpy as np
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode
import mani_skill.envs.tasks.tabletop.wall_bypass
import mani_skill.envs.tasks.tabletop.maze_drive

STEP = 0.5
ROT  = 0.3

GOOD_DIR = "demos/ContrastiveDemos/good"
BAD_DIR  = "demos/ContrastiveDemos/bad"

# create both output folders up front
os.makedirs(GOOD_DIR, exist_ok=True)
os.makedirs(BAD_DIR, exist_ok=True)

def make_keymap(step, rot):
    return {
        "w": [ step, 0, 0, 0, 0, 0, -1],
        "s": [-step, 0, 0, 0, 0, 0, -1],
        "a": [0,  step, 0, 0, 0, 0, -1],
        "d": [0, -step, 0, 0, 0, 0, -1],
        "q": [0, 0,  step, 0, 0, 0, -1],
        "e": [0, 0, -step, 0, 0, 0, -1],
        "i": [0, 0, 0,  rot, 0, 0, -1],
        "k": [0, 0, 0, -rot, 0, 0, -1],
        "j": [0, 0, 0, 0,  rot, 0, -1],
        "l": [0, 0, 0, 0, -rot, 0, -1],
        "u": [0, 0, 0, 0, 0,  rot, -1],
        "o": [0, 0, 0, 0, 0, -rot, -1],
        "f": [0, 0, 0, 0, 0, 0,  1],
        "g": [0, 0, 0, 0, 0, 0, -1],
        " ": [0, 0, 0, 0, 0, 0, -1],
    }

HELP = """
Controls (type key + Enter):
  Translation:  w/s = +/-x   a/d = +/-y   q/e = up/down
  Rotation:     i/k = +/-roll   j/l = +/-pitch   u/o = +/-yaw
  Gripper:      f = open   g = close
  Modes:        r = start/stop a BAD-demo snippet
  Other:        [space] = settle   +/- = step size   x = quit

Good demo -> saved to demos/MazeDrive-v1/good  (auto-saves on success)
Bad snippet -> saved to demos/MazeDrive-v1/bad  (press r to start, r to stop)
"""

def set_output_dir(env, output_dir):
    """Point the RecordEpisode wrapper at a new output directory."""
    os.makedirs(output_dir, exist_ok=True)
    env.output_dir = output_dir
    if hasattr(env, "_video_dir"):
        env._video_dir = output_dir
    if hasattr(env, "_trajectory_dir"):
        env._trajectory_dir = output_dir

def main():
    global STEP, ROT

    env = gym.make("WallBypass-v1", obs_mode="state",
                   control_mode="pd_ee_delta_pose",
                   render_mode="human", num_envs=1)
    # save_trajectory only; video is disabled (human render has no RGB frames)
    print("ENV ACTION SPACE:", env.action_space)
    env = RecordEpisode(env, output_dir=GOOD_DIR,
                        save_trajectory=True,
                        save_video=False,
                        save_on_reset=False,
                        source_type="human_teleop",
                        source_desc="teleop demo")

    env.reset(seed=0)
    print(HELP)

    keymap = make_keymap(STEP, ROT)
    recording_snippet = False   # True while capturing a bad snippet
    done = False

    def save_good():
        set_output_dir(env, GOOD_DIR)
        env.flush_trajectory()
        print(f">>> Saved GOOD demo -> {GOOD_DIR}")

    def save_bad():
        set_output_dir(env, BAD_DIR)
        env.flush_trajectory()
        print(f">>> Saved BAD snippet -> {BAD_DIR}")

    while not done:
        env.render()
        key = input("key> ").strip().lower()

        if key == "x":
            break

        if key == "+":
            STEP *= 1.5; ROT *= 1.5
            keymap = make_keymap(STEP, ROT)
            print(f"  step={STEP:.3f} rot={ROT:.3f}")
            continue
        if key == "-":
            STEP /= 1.5; ROT /= 1.5
            keymap = make_keymap(STEP, ROT)
            print(f"  step={STEP:.3f} rot={ROT:.3f}")
            continue

        # ---- bad-demo snippet toggle ----
        if key == "r":
            if not recording_snippet:
                recording_snippet = True
                print(">>> BAD snippet recording STARTED")
            else:
                save_bad()
                recording_snippet = False
                print(">>> BAD snippet recording STOPPED")
            continue
        if key not in keymap:
            print("  unknown key")
            continue

        obs, rew, term, trunc, info = env.step(
            np.array(keymap[key], dtype=np.float32))

        p = env.unwrapped.agent.tcp.pose.p[0].cpu().numpy()
        success = bool(info["success"][0])
        rec_flag = "BAD " if recording_snippet else "GOOD"
        print(f"  [{rec_flag}] TCP={p.round(3)} success={success}")

        # ---- GOOD demo: auto-save on success (only when NOT in snippet mode) ----
        if success and not recording_snippet:
            save_good()
            env.reset(seed=0)              # start a fresh good demo
            print(">>> New episode started. Play again or press x to quit.")

        if bool(trunc[0]):
            print(">>> Episode truncated (max steps). Resetting.")
            env.reset(seed=0)

    # Save whatever snippet is still open on quit
    if recording_snippet:
        save_bad()

    env.close()
    print("Exiting.")

if __name__ == "__main__":
    main()