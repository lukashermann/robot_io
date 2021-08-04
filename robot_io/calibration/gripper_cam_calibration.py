import math
import time

import hydra
import numpy as np

from robot_io.utils.utils import quat_to_euler, euler_to_quat
from robot_io.calibration.calibration import calibrate_gripper_cam_least_squares, visualize_calibration_gripper_cam
from robot_io.calibration.calibration import save_calibration, calibrate_gripper_cam_peak_martin, calculate_error


class GripperCamPoseSampler:
    """ Sample random poses """
    def __init__(self,
                 initial_pos,
                 initial_orn,
                 theta_limits,
                 r_limits,
                 h_limits,
                 trans_limits,
                 yaw_limits,
                 pitch_limit,
                 roll_limit):
        """
        Poses are sampled with polar coordinates theta and r around initial_pos, which are then perturbed with a random
        positional and rotational offset
        :param initial_pos: TCP position around which poses are sampled
        :param initial_orn: TCP orientation around which poses are sampled
        :param theta_limits: Angle for polar coordinate sampling wrt. X-axis in robot base frame
        :param r_limits: Radius for plar coordinate sampling
        :param h_limits: Sampling range for height offset
        :param trans_limits: Sampling range for lateral offset
        :param yaw_limits: Sampling range for yaw offset
        :param pitch_limit: Sampling range for pitch offset
        :param roll_limit: Sampling range for roll offset
        """
        self.initial_pos = np.array(initial_pos)
        self.initial_orn = quat_to_euler(np.array(initial_orn))
        self.theta_limits = theta_limits
        self.r_limits = r_limits
        self.h_limits = h_limits
        self.trans_limits = trans_limits
        self.yaw_limits = yaw_limits
        self.pitch_limit = pitch_limit
        self.roll_limit = roll_limit

    def sample_pose(self):
        """sample a  random pose"""
        theta = np.random.uniform(*self.theta_limits)
        vec = np.array([np.cos(theta), np.sin(theta), 0])
        vec = vec * np.random.uniform(*self.r_limits)
        yaw = np.random.uniform(*self.yaw_limits)
        trans = np.cross(np.array([0, 0, 1]), vec)
        trans = trans * np.random.uniform(*self.trans_limits)
        height = np.array([0, 0, 1]) * np.random.uniform(*self.h_limits)
        trans_final = self.initial_pos + vec + trans + height
        pitch = np.random.uniform(*self.pitch_limit)
        roll = np.random.uniform(*self.roll_limit)

        target_pos = np.array(trans_final)
        target_orn = np.array([math.pi + pitch, roll, theta + math.pi + yaw])
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
        time.sleep(0.3)
        marker_pose = marker_detector.estimate_pose()
        if marker_pose is not None:
            tcp_poses.append(robot.get_tcp_pose())
            marker_poses.append(marker_pose)
            i += 1

    return tcp_poses, marker_poses


@hydra.main(config_path="../conf", config_name="panda_calibrate_gripper_cam")
def main(cfg):
    cam = hydra.utils.instantiate(cfg.cam)
    marker_detector = hydra.utils.instantiate(cfg.marker_detector, cam=cam)
    robot = hydra.utils.instantiate(cfg.robot)
    tcp_poses, marker_poses = record_gripper_cam_trajectory(robot, marker_detector, cfg)
    # np.savez("data.npz", tcp_poses=tcp_poses, marker_poses=marker_poses)
    # data = np.load("data.npz")
    # tcp_poses = list(data["tcp_poses"])
    # marker_poses = list(data["marker_poses"])
    T_tcp_cam = calibrate_gripper_cam_least_squares(tcp_poses, marker_poses)

    visualize_calibration_gripper_cam(cam, T_tcp_cam)
    calculate_error(T_tcp_cam, tcp_poses, marker_poses)

    save_calibration(robot.name, cam.name, "cam", "tcp", T_tcp_cam)


if __name__ == "__main__":
    main()
