import time

import hydra
from robot_io.utils.utils import FpsController


@hydra.main(config_path="../conf")
def main(cfg):

    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)
    input_device = hydra.utils.instantiate(cfg.input, robot=robot)

    fps = FpsController(cfg.freq)

    obs = env.reset()
    # recorder = hydra.utils.instantiate(cfg.recorder)
    t1 = time.time()
    while True:
        action, record_info = input_device.get_action()
        obs, _, _, _ = env.step(action)
        # robot.visualize_joint_states()
        # recorder.step(action, obs, record_info)
        env.render()
        fps.step()
        # print(1 / (time.time() - t1))
        t1 = time.time()


if __name__ == "__main__":
    main()
