import time
from pathlib import Path

import hydra
import numpy as np

from robot_io.utils.utils import FpsController, angle_between_angles, quat_to_euler

N_DIGITS = 6

# def load_frames(path):
#     frames = sorted(list(Path(path).glob("frame*.npz")))
#     print(frames)
#     for frame in frames:
#         yield np.load(frame, allow_pickle=True)


def to_relative_action(prev_robot_state, robot_state, action):
    rel_pos = robot_state["tcp_pos"] - prev_robot_state["tcp_pos"]

    rel_orn = angle_between_angles(quat_to_euler(prev_robot_state["tcp_orn"]), quat_to_euler(robot_state["tcp_orn"]))

    gripper_action = action["motion"][-1]
    action = {"motion": (rel_pos, rel_orn, gripper_action), "ref": "rel"}
    return action

def to_absolute_action(robot_state, action):
    gripper_action = action["motion"][-1]
    action = {"motion": (robot_state["tcp_pos"], robot_state["tcp_orn"], gripper_action), "ref": "abs"}
    return action


def get_ep_start_end_ids(path):
    return np.sort(np.load(Path(path) / "ep_start_end_ids.npy"), axis=0)


def get_frame(path, i):
    filename = Path(path) / f"frame_{i:0{N_DIGITS}d}.npz"
    return np.load(filename, allow_pickle=True)


def reset(env, path, i):
    data = get_frame(path, i)
    robot_state = data['robot_state'].item()
    gripper_state = "open" if robot_state["gripper_opening_width"] > 0.07 else "closed"
    env.reset(target_pos=robot_state["tcp_pos"], target_orn=robot_state["tcp_orn"], gripper_state=gripper_state)


def get_action(path, i, use_rel_actions=False):
    frame = get_frame(path, i)
    robot_state = frame['robot_state'].item()
    recorded_action = frame["action"].item()
    if use_rel_actions:
        prev_robot_state = get_frame(path, i - 1)['robot_state'].item()
        action = to_relative_action(prev_robot_state, robot_state, recorded_action)
    else:
        action = to_absolute_action(robot_state, recorded_action)
    return action


@hydra.main(config_path="../conf", config_name="replay_recorded_trajectory")
def main(cfg):

    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    use_rel_actions = cfg.use_rel_actions
    ep_start_end_ids = get_ep_start_end_ids(cfg.load_dir)
    fps = FpsController(cfg.freq)

    obs = env.reset()

    for start_idx, end_idx in ep_start_end_ids:
        reset(env, cfg.load_dir, start_idx)
        for i in range(start_idx + 1, end_idx + 1):
            action = get_action(cfg.load_dir, i, use_rel_actions)
            print(action)
            obs, _, _, _ = env.step(action)
            env.render()
            fps.step()
            # print(1 / (time.time() - t1))
            t1 = time.time()


if __name__ == "__main__":
    main()
