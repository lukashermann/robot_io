import time
from pathlib import Path

import hydra
import numpy as np

from robot_io.utils.utils import FpsController


def load_frames(path):
    frames = sorted(list(Path(path).glob("frame*.npz")))
    print(frames)
    for frame in frames:
        yield np.load(frame, allow_pickle=True)


def get_action(frame):
    robot_state = frame['robot_state'].item()
    action = (robot_state["tcp_pos"], robot_state["tcp_orn"], 1 if robot_state["gripper_opening_width"] > 0.07 else -1)
    return action, frame["done"]


@hydra.main(config_path="../conf", config_name="replay_recorded_trajectory")
def main(cfg):

    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    # fps = FpsController(cfg.freq)

    obs = env.reset()

    for frame in load_frames(cfg.load_dir):
        action, done = get_action(frame)
        obs, _, _, _ = env.step(action)
        env.render()
        # fps.step()
        # print(1 / (time.time() - t1))
        t1 = time.time()


if __name__ == "__main__":
    main()
