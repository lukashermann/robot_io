import time

import rospy
import hydra
from panda_robot import PandaArm


def clean_shutdown():
    print("\nExiting example.")


@hydra.main(config_path="conf", config_name="panda_teleop")
def main(cfg):

    print("Initializing node... ")
    rospy.init_node("panda_vr_control_joint_impedance")
    print("Getting robot state... ")

    robot = PandaArm()
    input = hydra.utils.instantiate(cfg.input, robot=robot)
    panda_env = hydra.utils.instantiate(cfg.env, robot=robot)

    rospy.on_shutdown(clean_shutdown)
    rate = rospy.Rate(cfg.freq)

    obs = panda_env.reset()
    recorder = hydra.utils.instantiate(cfg.recorder)
    t1 = time.time()
    while not rospy.is_shutdown():
        action, record_info = input.get_action()
        obs, _, _, _ = panda_env.step(action)
        recorder.step(action, obs, record_info)
        panda_env.render()
        rate.sleep()
        # print(1 / (time.time() - t1))
        t1 = time.time()


if __name__ == "__main__":
    main()
