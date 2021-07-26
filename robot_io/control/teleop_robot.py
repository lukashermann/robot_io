import sys
import time

import hydra
from robot_io.utils.utils import FpsController


def wrap_train(config_name):
    """
    Wrapping hydra main such that you can load different configs from cmd line with
    python teleop_robot.py config_name=[kuka_teleop|panda_teleop]
    """
    @hydra.main(config_path="../conf", config_name=config_name)
    def main(cfg):

        robot = hydra.utils.instantiate(cfg.robot)
        input_device = hydra.utils.instantiate(cfg.input, robot=robot)
        env = hydra.utils.instantiate(cfg.env, robot=robot)

        fps = FpsController(cfg.freq)

        obs = env.reset()
        recorder = hydra.utils.instantiate(cfg.recorder)
        t1 = time.time()
        while True:
            action, record_info = input_device.get_action()
            obs, _, _, _ = env.step(action)
            recorder.step(action, obs, record_info)
            env.render()
            fps.step()
            # print(1 / (time.time() - t1))
            t1 = time.time()
    main()


def setup_config():
    config_str = next((x for x in sys.argv if "config_name" in x), None)
    if config_str is not None:
        config_name = config_str.split("=")[1]
        sys.argv.remove(config_str)
        return config_name
    else:
        return "panda_teleop"


if __name__ == "__main__":
    conf = setup_config()
    wrap_train(conf)
