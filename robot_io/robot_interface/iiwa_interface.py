import logging
import socket
import time

import numpy as np
from math import pi
from robot_io.robot_interface.wsg50_controller import WSG50Controller
from robot_io.robot_interface.base_robot_interface import BaseRobotInterface, GripperState
from robot_io.utils.utils import np_quat_to_scipy_quat, pos_orn_to_matrix, euler_to_quat, quat_to_euler, \
    matrix_to_pos_orn, xyz_to_zyx

import logging
log = logging.getLogger(__name__)

JAVA_JOINT_MODE = 0
JAVA_CARTESIAN_MODE_REL_PTP = 1
JAVA_CARTESIAN_MODE_ABS_PTP = 2
JAVA_CARTESIAN_MODE_REL_LIN = 3
JAVA_CARTESIAN_MODE_ABS_LIN = 4
JAVA_SET_PROPERTIES = 5
JAVA_GET_INFO = 6
JAVA_INIT = 7
JAVA_ABORT_MOTION = 8
JAVA_SET_FRAME = 9


# iiwa TCP frames
TCP_SHORT_FINGER = 20
TCP = 21


class IIWAInterface(BaseRobotInterface):
    def __init__(self,
                 host="localhost",
                 port=50100,
                 use_impedance=True,
                 joint_vel=0.1,
                 gripper_rot_vel=0.3,
                 joint_acc=0.3,
                 cartesian_vel=100,
                 cartesian_acc=300,
                 workspace_limits=((0.3, -0.3, 0.2), (0.6, 0.3, 0.4)),
                 tcp_name=TCP_SHORT_FINGER,
                 neutral_pose=(0.5, 0, 0.25, pi, 0, pi / 2)):
        """
        :param host: "localhost"
        :param port: default port is 50100
        :param use_impedance: Compliant robot. Check kuka docs before using.
        :param joint_vel: max velocities of joint 1-6, range [0, 1], for PTP/joint motions
        :param gripper_rot_vel: max velocities of joint 7, , range [0, 1], for PTP/joint motions
        :param joint_acc: max acceleration of joint 1-7, range [0,1], for PTP/joint motions
        :param cartesian_vel: max translational and rotational velocity of EE, in mm/s, for LIN motions
        :param cartesian_acc: max translational and rotational acceleration of EE, in mm/s**2, for LIN motions
        :param workspace_limits: Cartesian limits of TCP position, [x_min, x_max, y_min, y_max, z_min, z_max], in meter
        :param tcp_name: name of tcp frame in Java RoboticsAPI.data.xml
        """
        self.name = "iiwa"
        self.address = (host, port + 500)
        self.other_address = (host, port)
        self.version_counter = 1
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(self.address)
        self.socket.connect(self.other_address)
        self.use_impedance = use_impedance
        self._send_init_message()
        self.set_properties(joint_vel, gripper_rot_vel, joint_acc, cartesian_vel, cartesian_acc, use_impedance,
                            workspace_limits, tcp_name)
        self.neutral_pose = np.array(neutral_pose)
        self.gripper = WSG50Controller()
        self.gripper_state = GripperState.OPEN
        self.gripper.open_gripper()
        super().__init__()

    def move_to_neutral(self):
        if len(self.neutral_pose) == 6:
            target_pos = self.neutral_pose[:3]
            target_orn = euler_to_quat(self.neutral_pose[3:6])
            self.move_cart_pos_abs_ptp(target_pos, target_orn)
        elif len(self.neutral_pose) == 7:
            self.move_joint_pos(self.neutral_pose)

    def get_state(self):
        msg = np.array([self.version_counter], dtype=np.int32).tobytes()
        msg += np.array([JAVA_GET_INFO], dtype=np.int16).tobytes()
        state = self._send_recv_message(msg, 184)
        return self._create_info_dict(state)

    def get_tcp_pose(self):
        pos, orn = self.get_tcp_pos_orn()
        return pos_orn_to_matrix(pos, orn)

    def get_tcp_pos_orn(self):
        pos, orn = self.get_state()['tcp_pose'][0:3], self.get_state()['tcp_pose'][3:6]
        orn = euler_to_quat(orn)
        return pos, orn

    def move_cart_pos_abs_ptp(self, target_pos, target_orn):
        self.move_async_cart_pos_abs_ptp(target_pos, target_orn)
        while not self.reached_position(target_pos, target_orn):
            time.sleep(0.1)

    def move_joint_pos(self, joint_positions):
        self.move_async_joint_pos(joint_positions)
        while not self.reached_joint_state(joint_positions):
            time.sleep(0.1)

    def move_cart_pos_abs_lin(self, target_pos, target_orn):
        self.move_async_cart_pos_abs_lin(target_pos, target_orn)
        while not self.reached_position(target_pos, target_orn):
            time.sleep(0.1)

    def move_async_cart_pos_abs_ptp(self, target_pos, target_orn):
        pose = self._process_pose(target_pos, target_orn)
        msg = self._create_robot_msg(pose, JAVA_CARTESIAN_MODE_ABS_PTP)
        state = self._send_recv_message(msg, 184)

    def move_async_cart_pos_abs_lin(self, target_pos, target_orn):
        pose = self._process_pose(target_pos, target_orn)
        msg = self._create_robot_msg(pose, JAVA_CARTESIAN_MODE_ABS_LIN)
        state = self._send_recv_message(msg, 184)

    def move_async_cart_pos_rel_ptp(self, rel_target_pos, rel_target_orn):
        pose = self._process_pose(rel_target_pos, rel_target_orn)
        msg = self._create_robot_msg(pose, JAVA_CARTESIAN_MODE_REL_PTP)
        state = self._send_recv_message(msg, 184)

    def move_async_cart_pos_rel_lin(self, rel_target_pos, rel_target_orn):
        pose = self._process_pose(rel_target_pos, rel_target_orn)
        msg = self._create_robot_msg(pose, JAVA_CARTESIAN_MODE_REL_LIN)
        state = self._send_recv_message(msg, 184)

    def move_async_joint_pos(self, joint_positions):
        assert len(joint_positions) == 7
        joint_positions = np.array(joint_positions, dtype=np.float64)
        msg = self._create_robot_msg(joint_positions, JAVA_JOINT_MODE)
        state = self._send_recv_message(msg, 184)

    def abort_motion(self):
        msg = np.array([self.version_counter], dtype=np.int32).tobytes()
        msg += np.array([JAVA_ABORT_MOTION], dtype=np.int16).tobytes()
        return self._send_recv_message(msg, 188)

    def open_gripper(self, blocking=False):
        if self.gripper_state == GripperState.CLOSED:
            self.gripper.open_gripper()
            if blocking:
                # TODO: implement this properly
                time.sleep(1)
            self.gripper_state = GripperState.OPEN

    def close_gripper(self, blocking=False):
        if self.gripper_state == GripperState.OPEN:
            self.gripper.close_gripper()
            if blocking:
                # TODO: implement this properly
                time.sleep(1)
            self.gripper_state = GripperState.CLOSED

    @staticmethod
    def _create_info_dict(state):
        state[:3] *= 0.001
        return {'tcp_pose': state[:6], 'joint_positions': state[6:13], 'desired_tcp_pose': state[13:17],
                'force_torque': state[17:23]}

    def _send_recv_message(self, message, recv_msg_size):
        self.socket.send(bytes(message))
        reply, address = self.socket.recvfrom(4 * recv_msg_size)
        return np.frombuffer(reply, dtype=np.float64).copy()

    def _send_init_message(self):
        msg = np.array([self.version_counter], dtype=np.int32).tobytes()
        msg += np.array([JAVA_INIT], dtype=np.int16).tobytes()
        return self._send_recv_message(msg, 188)

    def set_properties(self, joint_vel, gripper_rot_vel, joint_acc, cartesian_vel, cartesian_acc, use_impedance,
                       workspace_limits, tcp_name):
        msg = np.array([self.version_counter], dtype=np.int32).tobytes()
        msg += np.array([JAVA_SET_PROPERTIES], dtype=np.int16).tobytes()
        msg += np.array([joint_vel], dtype=np.float64).tobytes()
        msg += np.array([gripper_rot_vel], dtype=np.float64).tobytes()
        msg += np.array([joint_acc], dtype=np.float64).tobytes()
        msg += np.array([cartesian_vel], dtype=np.float64).tobytes()
        msg += np.array([cartesian_acc], dtype=np.float64).tobytes()
        msg += np.array([use_impedance], dtype=np.int16).tobytes()
        msg += np.array([workspace_limits[0][0] * 1000], dtype=np.float64).tobytes()
        msg += np.array([workspace_limits[1][0] * 1000], dtype=np.float64).tobytes()
        msg += np.array([workspace_limits[0][1] * 1000], dtype=np.float64).tobytes()
        msg += np.array([workspace_limits[1][1] * 1000], dtype=np.float64).tobytes()
        msg += np.array([workspace_limits[0][2] * 1000], dtype=np.float64).tobytes()
        msg += np.array([workspace_limits[1][2] * 1000], dtype=np.float64).tobytes()
        msg += np.array([tcp_name], dtype=np.int16).tobytes()
        state = self._send_recv_message(msg, 188)

    def set_goal_frame(self, T_robot_goal, goal_workspace_limits):
        msg = np.array([self.version_counter], dtype=np.int32).tobytes()
        msg += np.array([JAVA_SET_FRAME], dtype=np.int16).tobytes()
        pos, orn = matrix_to_pos_orn(T_robot_goal)
        orn = xyz_to_zyx(quat_to_euler(orn))
        msg += np.array([pos[0] * 1000], dtype=np.float64).tobytes()
        msg += np.array([pos[1] * 1000], dtype=np.float64).tobytes()
        msg += np.array([pos[2] * 1000], dtype=np.float64).tobytes()
        msg += np.array([orn[0]], dtype=np.float64).tobytes()
        msg += np.array([orn[1]], dtype=np.float64).tobytes()
        msg += np.array([orn[2]], dtype=np.float64).tobytes()
        msg += np.array([goal_workspace_limits[0][0] * 1000], dtype=np.float64).tobytes()
        msg += np.array([goal_workspace_limits[1][0] * 1000], dtype=np.float64).tobytes()
        msg += np.array([goal_workspace_limits[0][1] * 1000], dtype=np.float64).tobytes()
        msg += np.array([goal_workspace_limits[1][1] * 1000], dtype=np.float64).tobytes()
        msg += np.array([goal_workspace_limits[0][2] * 1000], dtype=np.float64).tobytes()
        msg += np.array([goal_workspace_limits[1][2] * 1000], dtype=np.float64).tobytes()
        state = self._send_recv_message(msg, 188)

    @staticmethod
    def _process_pose(pos, orn):
        pos = np.array(pos, dtype=np.float64) * 1000
        orn = np.array(orn, dtype=np.float64)
        if len(orn) == 4:
            orn = quat_to_euler(orn)
        return np.concatenate([pos, orn])

    def _create_robot_msg(self, pose, mode):
        assert type(mode) == int
        msg = np.array([self.version_counter], dtype=np.int32).tobytes()
        msg += np.array([mode], dtype=np.int16).tobytes()
        for c in pose:
            msg += c.tobytes()
        msg += np.array([12345], dtype=np.int64).tobytes()
        return msg


if __name__ == "__main__":
    robot = IIWAInterface()
    robot.move_to_neutral()
    pos, orn = robot.get_tcp_pos_orn()
    pos[2] += 0.05
    robot.move_async_cart_pos_abs_ptp(pos, orn)

