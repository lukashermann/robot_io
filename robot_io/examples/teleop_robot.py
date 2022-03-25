import time

import hydra
from robot_io.utils.utils import FpsController


@hydra.main(config_path="../conf")
def main(cfg):
    """
    Teleoperate the robot with different input devices.
    Depending on the recorder, either record the whole interaction or only if the recording is triggered by the input
    device.

    Args:
        cfg: Hydra config
    """
    recorder = hydra.utils.instantiate(cfg.recorder)
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)
    env.reset()
    input_device = hydra.utils.instantiate(cfg.input, robot=robot)
    fps = FpsController(cfg.freq)
    t1 = time.time()

    while True:
        action, record_info = input_device.get_action()
        obs, _, _, _ = env.step(action)
        recorder.step(action, obs, record_info)
        env.render()
        fps.step()
        if cfg.show_fps:
            print(f"FPS: {1 / (time.time() - t1)}")
        t1 = time.time()


if __name__ == "__main__":
    main()
