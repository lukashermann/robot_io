import numpy as np
import cv2
import cv2.aruco as aruco
import math
import time
import random
from robot_io.robot_interface.iiwa_interface import IIWAInterface
from robot_io.cams.kinect4.kinect4 import Kinect4
from robot_io.calibration.marker_detection import detect_marker

from math import pi


from robot_io.cams.camera import Camera
from robot_io.robot_interface.base_robot_interface import BaseRobotInterface
from robot_io.utils.utils import matrix_to_pos_orn, \
                                 pos_orn_to_matrix, \
                                 pos3, \
                                 vec3, \
                                 inverse_frame

from scipy.spatial.transform import Rotation as R


# Uniform sampling of points on a sphere according to:
#  https://demonstrations.wolfram.com/RandomPointsOnASphere/
def np_sphere_sampling(n_points):
    r_theta = np.random.rand(n_points, 1) * np.pi
    r_u     = np.random.rand(n_points, 1)
    factor  = np.sqrt(1.0 - r_u**2)
    coords  = np.hstack((np.cos(r_theta) * factor, np.sin(r_theta) * factor, r_u))
    return coords # 

def np_random_quat_normal(n_rots, mean, std):
    return np.vstack([np.hstack((np.sin(r, ax), [np.cos(r)])) for ax, r in 
                                        zip(np_sphere_sampling(n_rots), np.random.normal(mean, std, n_rots))])

def np_random_normal_offset(n_points, mean, std):
    return np_sphere_sampling(n_points) * np.random.normal(mean, std, (n_points, 1))


def compute_residuals(x, ees_in_robot, obs_marker_in_cam):
    """Calculates the residuals of a camera and marker location estimate, given the observations.
    
    Args:
        x (list)             : Current pose estimate of robot relative to camera and marker relative to ee [x, y, z, rx, ry, rz, mx, my, mz]
        ees_in_robot (list)   : n Homogeneous transforms of end effector in robot frame
        obs_marker_in_cam (np.ndarray): n*4 Homogeneous points of marker in camera frame
    
    Returns:
        list: List of residuals
    """
    pos_marker_in_ee = pos3(*x[6:])
    cam_in_robot     = pos_orn_to_matrix(*x[:6])

    residuals = []

    for ee_in_robot, obs_pos_marker_in_cam in zip(ees_in_robot, obs_marker_in_cam):
        proj_pos_marker_in_cam = cam_in_robot.dot(ee_in_robot.dot(pos_marker_in_ee))
        residuals             += list(obs_pos_marker_in_cam - proj_pos_marker_in_cam)
    return residuals


def estimate_static_camera_pose(current_camera_estimate : np.ndarray,
                                current_marker_estimate : np.ndarray, 
                                ee_poses_in_robot       : List[np.ndarray],
                                marker_locations_in_cam : np.ndarray):
    """Estimates the best camera location and marker offset given a number of observations.
    
    Args:
        current_camera_estimate (np.ndarray): Current estimate of camera pose as [rx, ry, rz, x, y, z] in the robot frame
        current_marker_estimate (np.ndarray): Current estimate of marker offset to ee [x, y, z]
        eef_poses (List[np.ndarray]): List of poses that the ee has been commanded to in robot frame
        marker_locations_in_cam (np.ndarray): Observed locations of the marker in camera frame
    
    Returns:
        (np.ndarray, nd.array): New camera pose as [rx, ry, rz, x, y, z], new marker pose as [x, y, z]
    """
    x0 = np.hstack((current_camera_estimate, current_marker_estimate))
    result = least_squares(fun=compute_residuals, x0=x0, method='lm', args=(ee_poses_in_robot, marker_locations_in_cam))

    return result[:6], result[6:]


def calibration_behavior(robot : BaseRobotInterface,
                         camera : Camera,
                         initial_cam_in_robot : np.ndarray, 
                         initial_marker_ee=np.zeros(3),
                         marker_size=0.08,
                         marker_id=None,
                         dist_coeffs=np.zeros(12),
                         marker_dict=aruco.DICT_4X4_250):
    error  = 1e9
    error_delta = -1e9

    pos_cam_in_robot, quat_cam_in_robot = pos_orn_to_matrix(initial_cam_in_robot)
    cam_in_robot_estimate     = np.hstack((pos_cam_in_robot, quat_to_euler(quat_cam_in_robot)))
    pos_marker_in_ee_estimate = initial_marker_ee

    ee_in_robot = robot.get_tcp_pose()

    detections  = []
    ee_poses    = []
    
    sampling_scale = np.diag([0.5, 0.4, 0.2])

    while error_delta < -1e-3:
        # Draw a new ee_pose sample
        cam_in_robot  = pos_orn_to_matrix(cam_in_robot_estimate[:3], cam_in_robot_estimate[3:])
        ee_in_cam     = np.dot(inverse_frame(cam_in_robot), ee_in_robot)
        pos_ee_in_cam = ee_in_cam[:, 3]
        # Used to sample an elipsoid

        for offset in np_sphere_sampling(200):
            ee_goal_pos  = cam_in_robot.dot(ee_in_cam + sampling_scale.dot(offset))

            # Check
            for pose in ee_poses:
                if np.sqrt(np.sum((pose[:, 3] - ee_goal_pos)**2)) < 0.07:
                    ee_goal_pos = None
                    break

            if ee_goal_pos is None:
                continue

            # Add rotation
            ee_goal_quat_offset = np_random_quat_normal(np.pi * 0.05, np.pi * 0.25)
            ee_rot = np.eye(4)
            ee_rot[:3, :3] = ee_in_robot[:3, :3]
            ee_goal_rot = np.dot(ee_rot, pos_orn_to_matrix((0, 0, 0), ee_goal_quat_offset))

            ee_goal_pose = np.eye(4)
            ee_goal_pose[:3, :3] = ee_goal_rot[:3, :3]
            ee_goal_pose[:, :3]  = ee_goal_pos             

            # IK-check
            ik_solution = ik_solver.solve(ee_goal_pose)
            if ik_solution is None:
                print('Failed to find IK solution for sample.')
                continue
        else:
            raise Exception('Unable to find new working sample')

        # Move to pose and observe
        robot.move_joint_pos(ik_solution)
        time.sleep(0.1) # Sleep for a moment so everything is settled

        rgb, _ = camera.get_image()
        detected, marker_in_cam = detect_marker(rgb, marker_size, camera.get_camera_matrix(), marker_id, dist_coeffs, marker_dict)

        ee_pose = robot.get_tcp_pose()

        if detected:
            detections.append(marker_in_cam[:3, 3])
            ee_poses.append(ee_pose)

            cam_in_robot_estimate, pos_marker_in_ee = estimate_static_camera_pose(cam_in_robot_estimate, 
                                                                                  pos_marker_in_ee,
                                                                                  ee_poses,
                                                                                  detections)
            # Calculate linear error across all observations
            residuals = np.array(compute_residuals(np.hstack((cam_in_robot_estimate, pos_marker_in_ee)), 
                                                   ee_poses_in_robot, 
                                                   marker_locations_in_cam)).reshape((len(ee_poses), 3))
            lin_error   = np.sqrt(np.sum(residuals**2, axis=1))
            error_delta = np.mean(lin_error) - error
            error       = np.mean(lin_error)
        else:
            print('Failed to detect the marker at the current pose. '
                  'Driving back to a previous pose and trying again')

            goal_pose = random.choice(ee_poses)

            ik_solution = ik_solver.solve(goal_pose)
            robot.move_joint_pos(ik_solution)
            time.sleep(0.1)
            ee_pose = robot.get_tcp_pose()




def work_position(controller):
    pos = [0, -0.56, 0.26]
    orn = [pi, 0, pi / 2]
    controller.move_async_cart_pos_abs_ptp(pos, orn)


if __name__ == "__main__":
    cam = Kinect4()
    iiwa = IIWAInterface(use_impedance=True)
    # work_position(iiwa)
    max_angles = [165, 115, 165, 115, 165, 115, 170]
    min_angles = [-x for x in max_angles]
    dcm_list = []
    reset = True
    detect_counter = 0
    NUM_JOINTS = 7
    res_angles = np.zeros(NUM_JOINTS)
    last_joint = NUM_JOINTS - 1
    shift_degree = 10

    joint_angles = np.rad2deg(iiwa.get_state()['joint_positions'])
    if joint_angles[last_joint] < 0:
        res_angles[last_joint] = shift_degree
    else:
        res_angles[last_joint] = -shift_degree

    while True:
        rgb, depth = cam.get_image()
        detected, dcm = detect_marker(rgb, 0.1, marker_id=0, camera_matrix=cam.get_camera_matrix(),
                                    marker_dict=aruco.DICT_4X4_250)
        joint_angles = np.rad2deg(iiwa.get_state()['joint_positions'])

        # Detect the marker two times with small end-effector pose change -> Marker is discovered!
        if detected:

            detect_counter += 1
            if detect_counter > 2:
                print("Marker is discovered")
                print("dcm", dcm)
                exit()

            next_joint_angles = joint_angles + res_angles

        else:
            # Rotate end-effector to the closest extreme
            if reset or detect_counter != 0:
                reset = False
                detect_counter = 0
                if joint_angles[last_joint] < 0:
                    joint_angles[last_joint] = min_angles[last_joint]
                    res_angles[last_joint] = shift_degree
                else:
                    joint_angles[last_joint] = max_angles[last_joint]
                    res_angles[last_joint] = -shift_degree
                next_joint_angles = joint_angles

            # Rotate end-effector gradually to the other extreme
            else:
                next_joint_angles = joint_angles + res_angles

        next_joint_angles = np.clip(next_joint_angles, min_angles, max_angles).tolist()
        iiwa.move_async_joint_pos(np.deg2rad(next_joint_angles))
        time.sleep(3)







