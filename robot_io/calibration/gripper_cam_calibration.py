import math
import time

import hydra
import numpy as np

from robot_io.utils.utils import quat_to_euler, euler_to_quat
from robot_io.calibration.calibration import calibrate_gripper_cam_least_squares


class GripperCamPoseSampler:
    """ Sample random poses """
    def __init__(self,
                 initial_pos,
                 initial_orn,
                 theta_limits,
                 r_limits,
                 h_limits,
                 trans_limits,
                 rot_limits,
                 pitch_limit,
                 roll_limit):
        self.initial_pos = np.array(initial_pos)
        self.initial_orn = quat_to_euler(np.array(initial_orn))
        self.theta_limits = theta_limits
        self.r_limits = r_limits
        self.h_limits = h_limits
        self.trans_limits = trans_limits
        self.rot_limits = rot_limits
        self.pitch_limit = pitch_limit
        self.roll_limit = roll_limit

    def sample_pose(self):
        """sample a  random pose"""
        theta = np.random.uniform(*self.theta_limits)
        vec = np.array([np.cos(theta), np.sin(theta), 0])
        vec = vec * np.random.uniform(*self.r_limits)
        theta_offset = np.random.uniform(*self.rot_limits)
        trans = np.cross(np.array([0, 0, 1]), vec)
        trans = trans * np.random.uniform(*self.trans_limits)
        height = np.array([0, 0, 1]) * np.random.uniform(*self.h_limits)
        trans_final = self.initial_pos + vec + trans + height
        pitch = np.random.uniform(*self.pitch_limit)
        roll = np.random.uniform(*self.roll_limit)

        target_pos = np.array(trans_final)
        target_orn = np.array([math.pi + pitch, roll, theta + math.pi + theta_offset])
        target_orn = euler_to_quat(target_orn)
        return target_pos, target_orn


def record_gripper_cam_trajectory(robot, marker_detector, cfg):
    robot.move_to_neutral()
    _, orn = robot.get_tcp_pos_orn()
    pose_sampler = hydra.utils.instantiate(cfg.gripper_cam_pose_sampler, initial_orn=orn)

    i = 0
    tcp_poses = []
    marker_poses = []

    while i < cfg.num_poses:
        pos, orn = pose_sampler.sample_pose()
        robot.move_cart_pos_abs_ptp(pos, orn)
        time.sleep(0.5)
        detected, marker_pose = marker_detector.estimate_pose_for_marker_id(11)
        print(f"marker detected: {detected}")
        if detected:
            tcp_poses.append(robot.get_tcp_pose())
            marker_poses.append(marker_pose)
            i += 1

    return tcp_poses, marker_poses


@hydra.main(config_path="../conf")
def main(cfg):
    cam = hydra.utils.instantiate(cfg.cam)
    marker_detector = hydra.utils.instantiate(cfg.marker_detector, cam=cam)
    robot = hydra.utils.instantiate(cfg.robot)
    tcp_poses, marker_poses = record_gripper_cam_trajectory(robot, marker_detector, cfg)
    T_tcp_cam = calibrate_gripper_cam_least_squares(tcp_poses, marker_poses)
    print(T_tcp_cam)


if __name__ == "__main__":
    main()
