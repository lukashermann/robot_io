import time

import rospy
import numpy as np
from panda_robot import PandaArm
from franka_dataflow.getch import getch
from enum import Enum
import gym


class GripperState(Enum):
    OPEN = 1
    CLOSED = -1


class PandaEnv(gym.Env):
    def __init__(self, robot, force_threshold=10, torque_threshold=10, k_gains=0.25, d_gains=0.5, workspace_limits=None):
        """
        :param robot: instance of PandaArm class from panda_robot repository
        :param force_threshold: list of len 6 or scalar (gets repeated for all values)
        :param torque_threshold: list of len 7 or scalar (gets repeated for all values)
        :param k_gains: joint impedance k_gains
        :param d_gains: joint impedance d_gains
        :param workspace_limits: workspace bounding box [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        """
        if workspace_limits is None:
            workspace_limits = [[0.3, -0.5, 0.1], [0.7, 0.5, 0.5]]
        self.workspace_limits = workspace_limits
        self.robot = robot
        self.gripper = self.robot.get_gripper()
        assert self.gripper is not None
        self.gripper_state = GripperState.OPEN
        self.set_collision_threshold(force_threshold, torque_threshold)
        self.activate_impedance_controller(k_gains, d_gains)
        self.prev_j_des = self.robot._neutral_pose_joints

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
        pos, ori = self.robot.ee_pose()
        self.orn = ori

    def step(self, action):
        """
        Execute one action on the robot.
        :param action: cartesian action tuple position, orientation, gripper_action
        """
        assert isinstance(action, tuple) and len(action) == 3

        target_pos, target_orn, gripper_action = action
        target_pos = self.restrict_workspace(target_pos)
        status, j = self.robot.inverse_kinematics(target_pos, target_orn)
        if status:
            j_des = j
        else:
            print("Did not find IK Solution")
            j_des = self.prev_j_des
        self.robot.set_joint_positions_velocities(j_des, [0] * 7)  # impedance control command (see documentation at )
        self.prev_j_des = j_des

        if gripper_action == 1 and self.gripper_state == GripperState.CLOSED:
            self.gripper.move_joints(width=0.2, speed=3, wait_for_result=False)
            self.gripper_state = GripperState.OPEN
        elif gripper_action == -1 and self.gripper_state == GripperState.OPEN:
            self.gripper.grasp(width=0.02, force=5, speed=5, epsilon_inner=0.005, epsilon_outer=0.02,wait_for_result=False)
            self.gripper_state = GripperState.CLOSED

    def restrict_workspace(self, target_pos):
        return np.clip(target_pos, self.workspace_limits[0], self.workspace_limits[1])

    def render(self, mode='human'):
        pass