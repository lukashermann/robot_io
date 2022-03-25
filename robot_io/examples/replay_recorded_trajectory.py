import time
from pathlib import Path

import hydra
import numpy as np

from robot_io.utils.utils import FpsController, angle_between_angles, quat_to_euler

N_DIGITS = 6


def to_relative_action(prev_action, action):
    rel_pos = action["motion"][0] - prev_action["motion"][0]
    rel_orn = angle_between_angles(quat_to_euler(action["motion"][1] - prev_action["motion"][1]))

    gripper_action = action["motion"][-1]
    action = {"motion": (rel_pos, rel_orn, gripper_action), "ref": "rel"}
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
    action = frame["action"].item()
    if use_rel_actions:
        prev_action = get_frame(path, i - 1)['action'].item()
        return to_relative_action(prev_action, action)
    else:
        return action


@hydra.main(config_path="../conf", config_name="replay_recorded_trajectory")
def main(cfg):
    """
    Replay a recorded trajectory, either with absolute actions or relative actions.

    Args:
        cfg: Hydra config
    """
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    use_rel_actions = cfg.use_rel_actions
    ep_start_end_ids = get_ep_start_end_ids(cfg.load_dir)
    fps = FpsController(cfg.freq)

    env.reset()
    for start_idx, end_idx in ep_start_end_ids:
        reset(env, cfg.load_dir, start_idx)
        for i in range(start_idx + 1, end_idx + 1):
            action = get_action(cfg.load_dir, i, use_rel_actions)
            obs, _, _, _ = env.step(action)
            env.render()
            fps.step()


if __name__ == "__main__":
    main()
