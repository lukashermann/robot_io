import numpy as np
import cv2
import cv2.aruco as aruco
import math
import time
import random
import rospy

from math import pi
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R
from scipy.optimize          import least_squares

from kineverse_tools.ik_solver              import IKSolver
from kineverse.urdf_fix                     import load_urdf_file
from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer

from robot_io.robot_interface.iiwa_interface import IIWAInterface
from robot_io.cams.kinect4.kinect4 import Kinect4
from robot_io.calibration.marker_detection import detect_marker

from robot_io.cams.camera import Camera
from robot_io.robot_interface.base_robot_interface import BaseRobotInterface
from robot_io.utils.utils import matrix_to_pos_orn, \
                                 pos_orn_to_matrix, \
                                 pos3, \
                                 vec3, \
                                 inverse_frame, \
                                 quat_to_euler

# Uniform sampling of points on a sphere according to:
#  https://demonstrations.wolfram.com/RandomPointsOnASphere/
def np_sphere_sampling(n_points, points=False):
    r_theta = np.random.rand(n_points, 1) * np.pi
    r_u     = np.random.rand(n_points, 1)
    factor  = np.sqrt(1.0 - r_u**2)
    if points:
        return np.hstack((np.cos(r_theta) * factor, np.sin(r_theta) * factor, r_u, np.ones((n_points, 1))))
    return np.hstack((np.cos(r_theta) * factor, np.sin(r_theta) * factor, r_u, np.zeros((n_points, 1))))

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
    cam_in_robot     = pos_orn_to_matrix(x[:3], x[3:])

    residuals = []

    for ee_in_robot, obs_pos_marker_in_cam in zip(ees_in_robot, obs_marker_in_cam):
        proj_pos_marker_in_cam = cam_in_robot.dot(ee_in_robot.dot(pos_marker_in_ee))
        residuals             += list(obs_pos_marker_in_cam - proj_pos_marker_in_cam[:3])
    return residuals


def estimate_static_camera_pose(current_camera_estimate : np.ndarray,
                                current_marker_estimate : np.ndarray, 
                                ee_poses_in_robot       : list,
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

    return result.x[:6], result.x[6:]

def iiwa_6d_ik_wrapper(ik_solver, robot, goal_pose):
    q_now = {f'joint_a{x + 1}': v for x, v in enumerate(robot.get_state()['joint_positions'])}
    ee_goal_pos, ee_goal_quat = matrix_to_pos_orn(goal_pose)
    ik_error, ik_solution = ik_solver.solve(q_now, [ee_goal_pos[0], 
                                                    ee_goal_pos[1], 
                                                    ee_goal_pos[2],
                                                    ee_goal_quat.x,
                                                    ee_goal_quat.y,
                                                    ee_goal_quat.z,
                                                    ee_goal_quat.w])
    return ik_error, [v for k, v in sorted(ik_solution.items())]


def behavior_calibration(robot : BaseRobotInterface,
                         camera : Camera,
                         ik_solver,
                         initial_cam_in_robot : np.ndarray, 
                         initial_marker_ee=np.zeros(3),
                         marker_size=0.08,
                         marker_id=None,
                         dist_coeffs=np.zeros(12),
                         marker_dict=aruco.DICT_4X4_250,
                         visualizer=None):
    error  = 1e9
    error_delta = -1e9

    pos_cam_in_robot, quat_cam_in_robot = matrix_to_pos_orn(initial_cam_in_robot)
    cam_in_robot_estimate     = np.hstack((pos_cam_in_robot, quat_to_euler(quat_cam_in_robot)))
    pos_marker_in_ee          = initial_marker_ee

    initial_ee_pose = robot.get_tcp_pose()
    ee_in_robot     = initial_ee_pose
    trans_ee_in_robot       = np.eye(4)
    trans_ee_in_robot[:, 3] = ee_in_robot[:, 3]

    detections  = []
    ee_poses    = []

    # CHANGE SAMPLING AREA HERE
    sampling_scale = np.diag([0.3, 0.3, 0.3, 1])

    while error_delta < -1e-3 or error > 0.01:
        # Draw a new ee_pose sample
        cam_in_robot  = pos_orn_to_matrix(cam_in_robot_estimate[:3], cam_in_robot_estimate[3:])
        ee_in_cam     = np.dot(inverse_frame(cam_in_robot), ee_in_robot)
        trans_ee_in_cam = np.eye(4)
        trans_ee_in_cam[:, 3] = ee_in_cam[:, 3]
        # Used to sample an elipsoid

        sample_points       = np_random_normal_offset(200, 1.0, 0.3)
        sample_points[:, 3] = np.ones(sample_points.shape[0])
        cart_goal_points = trans_ee_in_robot.dot(sampling_scale.dot(sample_points.T))


        if visualizer is not None:
            visualizer.begin_draw_cycle('camera_estimate', 'ee_in_cam', 'ee_in_robot', 'sampled_locations')
            visualizer.draw_poses('camera_estimate', np.eye(4), 0.2, 0.01, [cam_in_robot])
            visualizer.draw_sphere('ee_in_cam', cam_in_robot.dot(ee_in_cam)[:, 3], 0.02)
            visualizer.draw_sphere('ee_in_robot', ee_in_robot[:, 3], 0.02, r=0, g=1)
            visualizer.draw_points('sampled_locations', np.eye(4), 0.02, cart_goal_points.T)
            visualizer.render('camera_estimate', 'ee_in_cam', 'ee_in_robot', 'sampled_locations')

        failed_samples = []

        for ee_goal_pos in cart_goal_points.T:
            # Check
            for pose in ee_poses:
                # Reject poses if they are below the table
                # ... if they are too close to the base
                # ... if they are too close to visited poses
                if ee_goal_pos[2] < 0.1 or \
                   np.sqrt(np.sum(ee_goal_pos[:2]**2)) < 0.4 or \
                   np.sqrt(np.sum((pose[:, 3] - ee_goal_pos)**2)) < 0.07:
                    failed_samples.append(ee_goal_pos)
                    ee_goal_pos = None
                    break

            if ee_goal_pos is None:
                continue

            # Add rotation
            ee_goal_quat_offset = np_random_quat_normal(1, np.pi * 0.05, np.pi * 0.25)[0]
            ee_rot = np.eye(4)
            ee_rot[:3, :3] = ee_in_robot[:3, :3]
            
            ee_goal_pose = np.eye(4)
            ee_goal_pose[:3, :3] = ee_in_robot[:3, :3] # np.dot(ee_rot, pos_orn_to_matrix((0, 0, 0), ee_goal_quat_offset))[:3, :3]
            ee_goal_pose[:,   3] = ee_goal_pos

            # IK-check
            ik_error, ik_solution = iiwa_6d_ik_wrapper(ik_solver, robot, ee_goal_pose)

            if ik_error > 0.01:
                print(f'Failed to find IK solution for sample. Error: {ik_error}')
                failed_samples.append(ee_goal_pos)
                continue
            break
        else:
            visualizer.begin_draw_cycle('failed_samples')
            visualizer.draw_points('failed_samples', np.eye(4), 0.02, failed_samples)
            visualizer.render('failed_samples')

            raise Exception('Unable to find new working sample')

        # Move to pose and observe
        robot.move_joint_pos(ik_solution) # THIS WORKS ONLY FOR KUKA
        time.sleep(0.1) # Sleep for a moment so everything is settled

        rgb, _ = camera.get_image()

        marker_in_cam = behavior_marker_seeker(robot, camera, 10, required_detections=1)

        ee_pose = robot.get_tcp_pose()

        if marker_in_cam is not None:
            detections.append(marker_in_cam[:3, 3])
            ee_poses.append(ee_pose)

            # Solver needs at least three detections
            if len(detections) < 3:
                continue

            cam_in_robot_estimate, pos_marker_in_ee = estimate_static_camera_pose(cam_in_robot_estimate, 
                                                                                  pos_marker_in_ee,
                                                                                  ee_poses,
                                                                                  detections)
            # Calculate linear error across all observations
            residuals = np.array(compute_residuals(np.hstack((cam_in_robot_estimate, pos_marker_in_ee)), 
                                                   ee_poses, 
                                                   detections)).reshape((len(ee_poses), 3))
            lin_error   = np.sqrt(np.sum(residuals**2, axis=1))
            error_delta = np.mean(lin_error) - error
            error       = np.mean(lin_error)

            print(f'Gathered new calibration sample.'
                  f'\n     Mean residual: {error}'
                  f'\n  Mean error delta: {error_delta}')
        else:
            print('Failed to detect the marker at the current pose. '
                  'Driving back to a previous pose and trying again')

            goal_pose = random.choice(ee_poses) if len(ee_poses) > 0 else initial_ee_pose

            ik_error, ik_solution = iiwa_6d_ik_wrapper(ik_solver, robot, ee_goal_pose)
            if ik_error >= 0.01:
                raise Exception(f'Failed to return to prior calibration pose. IK-Error: {ik_error}')
            robot.move_joint_pos(ik_solution)
            time.sleep(0.1)
            ee_pose = robot.get_tcp_pose()

    return pos_orn_to_matrix(cam_in_robot_estimate[:3], cam_in_robot_estimate[3:]), error


def behavior_marker_seeker(robot  : BaseRobotInterface, 
                           camera : Camera,
                           attempts=30,
                           required_detections=3):
    max_angles = [165, 115, 165, 115, 165, 115, 170]
    min_angles = [-x for x in max_angles]
    shift_degree = 10
    reset = True
    detect_counter = 0
    NUM_JOINTS = 7
    res_angles = np.zeros(NUM_JOINTS)

    joint_angles = np.rad2deg(iiwa.get_state()['joint_positions'])
    if joint_angles[-1] < 0:
        res_angles[-1] = shift_degree
    else:
        res_angles[-1] = -shift_degree

    for _ in tqdm(range(attempts), desc="Attempting to spot the marker"):
        rgb, depth    = camera.get_image()
        detected, dcm = detect_marker(rgb, 0.1, marker_id=0, 
                                      camera_matrix=camera.get_camera_matrix(),
                                      marker_dict=aruco.DICT_4X4_250)
        joint_angles = np.rad2deg(iiwa.get_state()['joint_positions'])

        # Detect the marker two times with small end-effector pose change -> Marker is discovered!
        if detected:

            detect_counter += 1
            if detect_counter >= required_detections:
                print("Marker is discovered")
                return dcm

            next_joint_angles = joint_angles + res_angles
        else:
            # Rotate end-effector to the closest extreme
            if reset or detect_counter != 0:
                reset = False
                detect_counter = 0
                if joint_angles[-1] < 0:
                    joint_angles[-1] = min_angles[-1]
                    res_angles[-1]   = shift_degree
                else:
                    joint_angles[-1] = max_angles[-1]
                    res_angles[-1]   = -shift_degree
                next_joint_angles    = joint_angles

            # Rotate end-effector gradually to the other extreme
            else:
                next_joint_angles = joint_angles + res_angles

        next_joint_angles = np.clip(next_joint_angles, min_angles, max_angles).tolist()
        iiwa.move_joint_pos(np.deg2rad(next_joint_angles))
        time.sleep(0.1)

    return None


def work_position(controller):
    pos = [0, -0.56, 0.26]
    orn = [pi, 0, pi / 2]
    controller.move_async_cart_pos_abs_ptp(pos, orn)


if __name__ == "__main__":
    cam  = Kinect4()
    iiwa = IIWAInterface(use_impedance=True)
    # work_position(iiwa)

    rospy.init_node('auto_calib')

    urdf_model = load_urdf_file('package://kineverse_experiment_world/urdf/iiwa_wsg_50.urdf')
    visualizer = ROSBPBVisualizer('~visualization')

    ik_solver  = IKSolver(urdf_model, 'wsg_50_tool_frame', visualizer)


    marker_in_cam = behavior_marker_seeker(iiwa, cam)

    if marker_in_cam is None:
        print('Failed to localize the marker.')
        exit()

    print(marker_in_cam)

    ee_pose = iiwa.get_tcp_pose()

    cam_in_robot, residual = behavior_calibration(iiwa,
                                                  cam,
                                                  ik_solver,
                                                  np.dot(ee_pose, inverse_frame(marker_in_cam)),
                                                  visualizer=visualizer)

    print(f'Calibrated camera pose:\n{cam_in_robot}\nCalibration residual: {residual}')




