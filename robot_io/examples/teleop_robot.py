import time

import hydra
from robot_io.utils.utils import FpsController


@hydra.main(config_path="../conf")
def main(cfg):
    recorder = hydra.utils.instantiate(cfg.recorder)
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)
    obs = env.reset()
    input_device = hydra.utils.instantiate(cfg.input, robot=robot)
    fps = FpsController(cfg.freq)
    t1 = time.time()

    while True:
        action, record_info = input_device.get_action()
        obs, _, _, _ = env.step(action)
        robot.visualize_joint_states()
        robot.visualize_external_forces()
        recorder.step(action, obs, record_info)
        env.render()
        fps.step()
        # print(1 / (time.time() - t1))
        t1 = time.time()


if __name__ == "__main__":
    main()
