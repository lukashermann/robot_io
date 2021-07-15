import time

import cv2
import rospy
import numpy as np
from panda_robot import PandaArm
from franka_dataflow.getch import getch
from enum import Enum
import gym
from robot_io.panda_control.IKfast_panda import IKfast
from robot_io.cams.framos.framos_d435e import FramosD435e
from robot_io.cams.kinect4.kinect4_threading import Kinect4
from copy import deepcopy


class GripperState(Enum):
    OPEN = 1
    CLOSED = -1


class PandaEnv(gym.Env):
    def __init__(self,
                 robot,
                 use_gripper_cam=True,
                 num_static_cams=1,
                 force_threshold=10,
                 torque_threshold=10,
                 k_gains=0.25,
                 d_gains=0.5,
                 workspace_limits=None,
                 ik_solver='kdl',
                 rest_pose=(-1.465, 1.481, 1.525, -2.435, -1.809, 1.855, -1.231)):
        """
        :param use_gripper_cam: bool
        :param num_static_cams: int
        :param robot: instance of PandaArm class from panda_robot repository
        :param force_threshold: list of len 6 or scalar (gets repeated for all values)
        :param torque_threshold: list of len 7 or scalar (gets repeated for all values)
        :param k_gains: joint impedance k_gains
        :param d_gains: joint impedance d_gains
        :param workspace_limits: workspace bounding box [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        :param ik_solver: kdl or ik_fast
        :param rest_pose: joint_positions for null space (only for ik_fast)
        """
        if workspace_limits is None:
            workspace_limits = [[0.3, -0.5, 0.1], [0.6, 0.5, 0.5]]
        self.workspace_limits = workspace_limits
        self.robot = robot
        self.gripper = self.robot.get_gripper()
        assert self.gripper is not None
        self.gripper_state = GripperState.OPEN
        self.set_collision_threshold(force_threshold, torque_threshold)
        self.activate_impedance_controller(k_gains, d_gains)
        self.prev_j_des = self.robot._neutral_pose_joints
        self.ik_solver = ik_solver
        if ik_solver == 'ik_fast':
            self.ik_fast = IKfast(rp=rest_pose, joint_limits=self.robot.joint_limits(), weights=(10, 8, 6, 6, 2, 2, 1), num_angles=50)
        self.gripper_cam = None
        if use_gripper_cam:
            self.gripper_cam = FramosD435e(img_type='rgb_depth')
        self.static_cams = None
        if num_static_cams > 0:
            self.static_cams = [Kinect4(device=i) for i in range(num_static_cams)]
        self.obs = None

    def set_collision_threshold(self, force_threshold, torque_threshold):
        """
        :param force_threshold: list of len 6 or scalar (gets repeated for all values)
        :param torque_threshold: list of len 7 or scalar (gets repeated for all values)
        """
        if isinstance(force_threshold, (int, float)):
            force_threshold = [force_threshold] * 6  # cartesian force threshold
        else:
            assert len(force_threshold) == 6
        if isinstance(torque_threshold, (int, float)):
            torque_threshold = [torque_threshold] * 7  # joint torque threshold
        else:
            assert len(torque_threshold) == 7
        self.robot.set_collision_threshold(joint_torques=torque_threshold, cartesian_forces=force_threshold)

    def activate_impedance_controller(self, k_gains, d_gains):
        """
        Activate joint impedance controller.
        :param k_gains: List of len 7 or scalar, which is interpreted as a scaling factor for default k_gains
        :param d_gains: List of len 7 or scalar, which is interpreted as a scaling factor for default k_gains
        """
        cm = self.robot.get_controller_manager()
        if not cm.is_running("franka_ros_interface/effort_joint_impedance_controller"):
            cm.stop_controller(cm.current_controller)
            cm.start_controller("franka_ros_interface/effort_joint_impedance_controller")
        time.sleep(1)

        default_k_gains = np.array([1200.0, 1000.0, 1000.0, 800.0, 300.0, 200.0, 50.0])
        default_d_gains = np.array([50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0])

        if isinstance(k_gains, (float, int)):
            assert 0.2 < k_gains <= 1
            k_gains = list(default_k_gains * k_gains)
        elif k_gains is None:
            k_gains = list(default_k_gains)

        if isinstance(d_gains, (float, int)):
            assert 0.5 <= d_gains <= 1
            d_gains = list(default_d_gains * d_gains)
        elif d_gains is None:
            d_gains = list(default_d_gains)

        assert len(k_gains) == 7
        assert len(d_gains) == 7

        ctrl_cfg_client = cm.get_current_controller_config_client()
        ctrl_cfg_client.set_controller_gains(k_gains, d_gains)

    def reset(self):
        """
        Reset robot to neutral position.
        """
        self.robot.move_to_neutral()
        return self._get_obs()

    def _inverse_kinematics(self, target_pos, target_orn):
        """
        :param target_pos: cartesian target position
        :param target_orn: cartesian target orientation
        :return: status (True if solution was found), target_joint_positions
        """
        if self.ik_solver == 'kdl':
            status, j = self.robot.inverse_kinematics(target_pos, target_orn)
        elif self.ik_solver == 'ik_fast':
            status, j = self.ik_fast.inverse_kinematics(target_pos, target_orn)
        else:
            raise NotImplementedError

        if status:
            j_des = j
        else:
            print("Did not find IK Solution")
            j_des = self.prev_j_des
        self.prev_j_des = j_des
        return j_des

    def _get_obs(self):
        obs = {}
        if self.gripper_cam is not None:
            rgb_gripper, depth_gripper = self.gripper_cam.get_image()
            obs['rgb_gripper'] = rgb_gripper
            obs['depth_gripper'] = depth_gripper
        if self.static_cams is not None:
            for i, cam in enumerate(self.static_cams):
                rgb, depth = cam.get_image()
                obs[f'rgb_static_{i}'] = rgb
                obs[f'depth_static_{i}'] = depth

        obs['robot_state'] = deepcopy(self.robot.state())
        self.obs = obs
        return obs

    def step(self, action):
        """
        Execute one action on the robot.
        :param action: cartesian action tuple position, orientation, gripper_action
        :return: obs, reward, done, info
        """
        if action is None:
            return self._get_obs(), 0, False, {}
        assert isinstance(action, tuple) and len(action) == 3

        target_pos, target_orn, gripper_action = action
        target_pos = self._restrict_workspace(target_pos)
        j_des = self._inverse_kinematics(target_pos, target_orn)

        self.robot.set_joint_positions_velocities(j_des, [0] * 7)  # impedance control command (see documentation at )

        if gripper_action == 1 and self.gripper_state == GripperState.CLOSED:
            self.gripper.move_joints(width=0.2, speed=3, wait_for_result=False)
            self.gripper_state = GripperState.OPEN
        elif gripper_action == -1 and self.gripper_state == GripperState.OPEN:
            self.gripper.grasp(width=0.02, force=5, speed=5, epsilon_inner=0.005, epsilon_outer=0.02,wait_for_result=False)
            self.gripper_state = GripperState.CLOSED

        obs = self._get_obs()
        return obs, 0, False, {}

    def _restrict_workspace(self, target_pos):
        """
        :param target_pos: cartesian target position
        :return: clip target_pos at workspace limits
        """
        return np.clip(target_pos, self.workspace_limits[0], self.workspace_limits[1])

    def render(self, mode='human'):
        if mode == 'human' and self.obs is not None:
            if "rgb_gripper" in self.obs:
                cv2.imshow("rgb_gripper", self.obs["rgb_gripper"][:, :, ::-1])
            if "rgb_static_0" in self.obs:
                cv2.imshow("rgb_static_0", self.obs["rgb_static_0"][:, :, ::-1])
            if "rgb_static_1" in self.obs:
                cv2.imshow("rgb_static_1", self.obs["rgb_static_1"][:, :, ::-1])
            cv2.waitKey(1)
