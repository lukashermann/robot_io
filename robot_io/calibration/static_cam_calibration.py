import time
import os
from pathlib import Path

import hydra
import numpy as np

from robot_io.utils.utils import matrix_to_pos_orn
from robot_io.calibration.calibration import calibrate_static_cam_least_squares, visualize_frame_in_static_cam
from robot_io.calibration.calibration import save_calibration, calculate_error
from robot_io.utils.utils import FpsController


def record_static_cam_trajectory(robot, marker_detector, cfg, calib_poses_dir):

    input_device = hydra.utils.instantiate(cfg.input, robot=robot)
    fps = FpsController(cfg.freq)

    robot.move_to_neutral()
    time.sleep(5)
    robot.close_gripper(blocking=True)

    tcp_poses = []
    marker_poses = []

    recorder = hydra.utils.instantiate(cfg.recorder, save_dir=calib_poses_dir)
    while True:
        fps.step()
        action, record_info = input_device.get_action()
        if record_info["hold_event"]:
            return tcp_poses, marker_poses

        if action is None:
            continue

        target_pos, target_orn, _ = action['motion']
        # TODO: use LIN for panda
        robot.move_async_cart_pos_abs_ptp(target_pos, target_orn)

        marker_pose = marker_detector.estimate_pose()
        if marker_pose is not None:
            tcp_pose = robot.get_tcp_pose()
            tcp_poses.append(tcp_pose)
            marker_poses.append(marker_pose)
            recorder.step(tcp_pose, marker_pose, record_info)


def detect_marker_from_trajectory(robot, tcp_poses, marker_detector, cfg):

    robot.move_to_neutral()
    time.sleep(5)
    robot.close_gripper(blocking=True)
    marker_poses = []
    valid_tcp_poses = []

    record_info = {"trigger_release": True}
    for i in range(len(tcp_poses)):
        target_pos, target_orn = matrix_to_pos_orn(tcp_poses[i])
        robot.move_cart_pos_abs_ptp(target_pos, target_orn)
        time.sleep(1.0)
        marker_pose = marker_detector.estimate_pose()
        if marker_pose is not None:
            valid_tcp_poses.append(tcp_poses[i])
            marker_poses.append(marker_pose)

    return valid_tcp_poses, marker_poses


def load_static_cam_trajectory(path):

    tcp_poses = []
    marker_poses = []

    for filename in path.glob("*.npz"):
        pose = np.load(filename)
        tcp_poses.append(pose['tcp_pose'])
        marker_poses.append(pose['marker_pose'])

    return tcp_poses, marker_poses


@hydra.main(config_path="../conf")
def main(cfg):
    cam = hydra.utils.instantiate(cfg.cam)
    marker_detector = hydra.utils.instantiate(cfg.marker_detector, cam=cam)
    robot = hydra.utils.instantiate(cfg.robot)
    calib_poses_dir = Path(f"{robot.name}_{marker_detector.cam.name}_calib_poses")
    if cfg.record_traj:
        tcp_poses, marker_poses = record_static_cam_trajectory(robot, marker_detector, cfg, calib_poses_dir)
    elif cfg.play_traj:
        tcp_poses, _ = load_static_cam_trajectory(calib_poses_dir)
        tcp_poses, marker_poses = detect_marker_from_trajectory(robot, tcp_poses, marker_detector, cfg)
    else:
        tcp_poses, marker_poses = load_static_cam_trajectory(calib_poses_dir)
    T_robot_cam = calibrate_static_cam_least_squares(tcp_poses, marker_poses)
    save_calibration(robot.name, cam.name, "cam", "robot", T_robot_cam)
    # calculate_error(T_tcp_cam, tcp_poses, marker_poses)
    # print(T_robot_cam)
    visualize_frame_in_static_cam(cam, np.linalg.inv(T_robot_cam))

if __name__ == "__main__":
    main()
