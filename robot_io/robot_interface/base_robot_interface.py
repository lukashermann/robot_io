from enum import Enum

import cv2
from scipy.spatial.transform.rotation import Rotation as R
import numpy as np

from robot_io.utils.utils import euler_to_quat


class GripperState(Enum):
    OPEN = 1
    CLOSED = -1


class BaseRobotInterface:
    """
    Generic interface for robot control.
    Not all methods must be implemented.
    """
    def __init__(self, ll, ul, *args, **kwargs):
        self.ll = np.array(ll)
        self.ul = np.array(ul)

    def move_to_neutral(self):
        """
        Move robot to initial position defined in robot conf. This method is blocking.
        """
        raise NotImplementedError

    def get_state(self):
        """
        :return: Dictionary with full robot state
        """
        raise NotImplementedError

    def get_tcp_pose(self):
        """
        :return: Tcp pose as homogeneous matrix (4x4 np.ndarray)
        """
        raise NotImplementedError

    def get_tcp_pos_orn(self):
        """
        :return: Tcp pose as tuple (pos (x,y,z), orn (x,y,z,w))
        """
        raise NotImplementedError

    def move_cart_pos_abs_ptp(self, target_pos, target_orn):
        """
        Move robot to absolute cartesian pose with a PTP motion, blocking
        :param target_pos: (x,y,z)
        :param target_orn: quaternion (x,y,z,w) | euler_angles (x,y,z)
        """
        raise NotImplementedError

    def move_joint_pos(self, joint_positions):
        """
        Move robot to absolute joint positions, blocking.
        :param joint_positions: (j1, ..., jn)
        """
        raise NotImplementedError

    def move_async_cart_pos_abs_ptp(self, target_pos, target_orn):
        """
        Move robot to absolute cartesian pose with a PTP motion, non blocking
        :param target_pos: (x,y,z)
        :param target_orn: quaternion (x,y,z,w) | euler_angles (x,y,z)
        """
        raise NotImplementedError

    def move_async_cart_pos_abs_lin(self, target_pos, target_orn):
        """
        Move robot to absolute cartesian pose with a LIN motion, non blocking
        :param target_pos: (x,y,z)
        :param target_orn: quaternion (x,y,z,w) | euler_angles (x,y,z)
        """
        raise NotImplementedError

    def move_async_cart_pos_rel_ptp(self, rel_target_pos, rel_target_orn):
        """
        Move robot to relative cartesian pose with a PTP motion, non blocking
        :param rel_target_pos: position offset (x,y,z)
        :param rel_target_orn: orientation offset quaternion (x,y,z,w) | euler_angles (x,y,z)
        """
        raise NotImplementedError

    def move_async_cart_pos_rel_lin(self, rel_target_pos, rel_target_orn):
        """
        Move robot to relative cartesian pose with a LIN motion, non blocking
        :param rel_target_pos: position offset (x,y,z)
        :param rel_target_orn: orientation offset quaternion (x,y,z,w) | euler_angles (x,y,z)
        """
        raise NotImplementedError

    def move_async_joint_pos(self, joint_positions):
        """
        Move robot to absolute joint positions, non blocking.
        :param joint_positions: (j1, ..., jn)
        """
        raise NotImplementedError

    def move_async_joint_vel(self, joint_velocities):
        """
        Move robot with joint velocities, non blocking.
        :param joint_velocities: (v1, ..., vn)
        """
        raise NotImplementedError

    def abort_motion(self):
        """
        Stop the execution of the current motion.
        """
        raise NotImplementedError

    def close_gripper(self, blocking=False):
        """
        Close fingers of the gripper.
        :param blocking: wait for gripper action to be finished
        """
        raise NotImplementedError

    def open_gripper(self, blocking=False):
        """
        Open fingers of the gripper.
        :param blocking: wait for gripper action to be finished
        """
        raise NotImplementedError

    def reached_position(self, target_pos, target_orn, cart_threshold=0.005, orn_threshold=0.05):
        """
        Check if robot has reached a target pose
        :param target_pos: (x,y,z)
        :param target_orn: quaternion (x,y,z,w) | euler_angles (x,y,z)
        :param cart_threshold: cartesian position error threshold for euclidean distance
                               between current_pos and target_pos, in meter
        :param orn_threshold: angular error threshold, in radian.
        :return: True if reached pose, else False
        """
        if len(target_orn) == 3:
            target_orn = euler_to_quat(target_orn)
        curr_pos, curr_orn = self.get_tcp_pos_orn()
        pos_error = np.linalg.norm(target_pos - curr_pos)
        orn_error = np.linalg.norm((R.from_quat(target_orn) * R.from_quat(curr_orn).inv()).as_rotvec())
        return pos_error < cart_threshold and orn_error < orn_threshold

    def reached_joint_state(self, target_state, threshold=0.001):
        """
        Check if robot has reached a target joint state
        :param target_state: (j1, ..., jn)
        :param threshold:
        :return: True if reached state, else False
        """
        curr_pos = self.get_state()['joint_positions']
        offset = np.sum(np.abs((np.array(target_state) - curr_pos)))
        return offset < threshold

    def visualize_joint_states(self):
        canvas = np.ones((300, 300, 3))
        joint_states = self.get_state()["joint_positions"]
        left = 10
        right = 290
        width = right - left
        height = 30
        y = 10
        for i, (l, q, u) in enumerate(zip(self.ll, joint_states, self.ul)):
            cv2.rectangle(canvas, [left, y], [right, y + height], [0,0,0], thickness=2)
            bar_pos = int(left + width * (q - l) / (u - l))
            cv2.line(canvas, [bar_pos, y], [bar_pos, y + height], thickness=5, color=[0, 0, 1])
            y += height + 10
        cv2.imshow("joint_positions", canvas)
        cv2.waitKey(1)
