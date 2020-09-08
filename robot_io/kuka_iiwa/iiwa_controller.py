import socket
import numpy as np
import time
import argparse
from math import pi
from scipy.spatial.transform import Rotation as R

JAVA_JOINT_MODE = 0
JAVA_CARTESIAN_MODE_REL_PTP = 1
JAVA_CARTESIAN_MODE_ABS_PTP = 2
JAVA_CARTESIAN_MODE_REL_LIN = 3
JAVA_CARTESIAN_MODE_ABS_LIN = 4
JAVA_SET_PROPERTIES = 5
JAVA_GET_INFO = 6
JAVA_INIT = 7
JAVA_ABORT_MOTION = 8


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


class IIWAController:

    def __init__(self,
                 host="localhost",
                 port=50100,
                 use_impedance=True,
                 joint_vel=0.1,
                 gripper_rot_vel=0.3,
                 joint_acc=0.3,
                 cartesian_vel=100,
                 cartesian_acc=300,
                 workspace_limits=(-0.25, 0.25, -0.75, -0.41, 0.13, 0.3)):
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

        """
        self.address = (host, port + 500)
        self.other_address = (host, port)
        self.version_counter = 1
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(self.address)
        self.socket.connect(self.other_address)
        self.use_impedance = use_impedance
        self._send_init_message()
        self.set_properties(joint_vel, gripper_rot_vel, joint_acc, cartesian_vel, cartesian_acc, use_impedance, workspace_limits)

    def _send_recv_message(self, message, recv_msg_size):
        self.socket.send(bytes(message))
        reply, address = self.socket.recvfrom(4 * recv_msg_size)
        return np.frombuffer(reply, dtype=np.float64).copy()

    def _send_init_message(self):
        msg = np.array([self.version_counter], dtype=np.int32).tostring()
        msg += np.array([JAVA_INIT], dtype=np.int16).tostring()
        return self._send_recv_message(msg, 188)

    def set_properties(self, joint_vel, gripper_rot_vel, joint_acc, cartesian_vel, cartesian_acc, use_impedance, workspace_limits):
        msg = np.array([self.version_counter], dtype=np.int32).tostring()
        msg += np.array([JAVA_SET_PROPERTIES], dtype=np.int16).tostring()
        msg += np.array([joint_vel], dtype=np.float64).tostring()
        msg += np.array([gripper_rot_vel], dtype=np.float64).tostring()
        msg += np.array([joint_acc], dtype=np.float64).tostring()
        msg += np.array([cartesian_vel], dtype=np.float64).tostring()
        msg += np.array([cartesian_acc], dtype=np.float64).tostring()
        msg += np.array([use_impedance], dtype=np.int16).tostring()
        msg += np.array([workspace_limits[0] * 1000], dtype=np.float64).tostring()
        msg += np.array([workspace_limits[1] * 1000], dtype=np.float64).tostring()
        msg += np.array([workspace_limits[2] * 1000], dtype=np.float64).tostring()
        msg += np.array([workspace_limits[3] * 1000], dtype=np.float64).tostring()
        msg += np.array([workspace_limits[4] * 1000], dtype=np.float64).tostring()
        msg += np.array([workspace_limits[5] * 1000], dtype=np.float64).tostring()
        state = self._send_recv_message(msg, 188)
        return self._create_info_dict(state)

    def _create_robot_msg(self, coord, mode):
        assert (type(mode) == int)
        msg = np.array([self.version_counter], dtype=np.int32).tostring()
        msg += np.array([mode], dtype=np.int16).tostring()
        for c in coord:
            msg += np.array([c], dtype=np.float64).tostring()
        msg += np.array([12345], dtype=np.int64).tostring()
        return msg

    def get_info(self):
        msg = np.array([self.version_counter], dtype=np.int32).tostring()
        msg += np.array([JAVA_GET_INFO], dtype=np.int16).tostring()
        state = self._send_recv_message(msg, 184)
        return self._create_info_dict(state)

    def get_tcp_pose(self):
        return self.get_info()['tcp_pose']

    def _create_info_dict(self, state):
        state[:3] *= 0.001
        return {'tcp_pose': state[:6], 'joint_positions': state[6:13], 'desired_tcp_pose': state[13:17],
                'force_torque': state[17:23]}

    def send_joint_angles(self, joint_angles, mode="degrees"):
        assert (type(joint_angles) == tuple)
        assert (len(joint_angles) == 7)
        coord = np.array(joint_angles, dtype=np.float64)
        if mode == 'degrees':
            coord = (coord / 180 * pi).tolist()
        else:
            coord = coord.tolist()
        msg = self._create_robot_msg(coord, JAVA_JOINT_MODE)
        print("sending joint angles")
        state = self._send_recv_message(msg, 184)
        return self._create_info_dict(state)

    def send_joint_angles_rad(self, joint_angles):
        assert (type(joint_angles) == tuple)
        assert (len(joint_angles) == 7)
        coord = np.array(joint_angles, dtype=np.float64)
        msg = self._create_robot_msg(coord, JAVA_JOINT_MODE)
        print("sending joint angles")
        state = self._send_recv_message(msg, 184)
        return self._create_info_dict(state)

    def send_cartesian_coords_rel_PTP(self, coords):
        assert (type(coords) == tuple)
        assert (len(coords) == 6)
        coord = np.array(coords, dtype=np.float64)
        coord[:3] *= 1000
        coord = coord.tolist()
        msg = self._create_robot_msg(coord, JAVA_CARTESIAN_MODE_REL_PTP)
        state = self._send_recv_message(msg, 184)
        return self._create_info_dict(state)

    @timeit
    def send_cartesian_coords_abs_PTP(self, coords):
        assert (type(coords) == tuple)
        assert (len(coords) == 6)
        coord = np.array(coords, dtype=np.float64)
        coord[:3] *= 1000
        coord = coord.tolist()
        msg = self._create_robot_msg(coord, JAVA_CARTESIAN_MODE_ABS_PTP)
        state = self._send_recv_message(msg, 184)
        return self._create_info_dict(state)

    @timeit
    def send_cartesian_coords_rel_LIN(self, coords):
        assert (type(coords) == tuple)
        assert (len(coords) == 6)
        coord = np.array(coords, dtype=np.float64)
        coord[:3] *= 1000
        coord = coord.tolist()
        msg = self._create_robot_msg(coord, JAVA_CARTESIAN_MODE_REL_LIN)
        state = self._send_recv_message(msg, 184)
        return self._create_info_dict(state)

    @timeit
    def send_cartesian_coords_abs_LIN(self, coords):
        assert (type(coords) == tuple)
        assert (len(coords) == 6)
        coord = np.array(coords, dtype=np.float64)
        coord[:3] *= 1000
        coord = coord.tolist()
        msg = self._create_robot_msg(coord, JAVA_CARTESIAN_MODE_ABS_LIN)
        state = self._send_recv_message(msg, 184)
        return self._create_info_dict(state)

    def abort_motion(self):
        msg = np.array([self.version_counter], dtype=np.int32).tostring()
        msg += np.array([JAVA_ABORT_MOTION], dtype=np.int16).tostring()
        return self._send_recv_message(msg, 188)

    def reached_position(self, pos):
        cart_threshold = 0.005 if self.use_impedance else 0.001
        or_threshold = 0.05 if self.use_impedance else 0.001
        curr_pos = self.get_tcp_pose()
        cart_offset = np.linalg.norm(np.array(pos)[:3] - curr_pos[:3])
        or_offset = np.sum(np.abs((R.from_dcm(R.from_euler('xyz', pos[3:]).as_dcm() @ np.linalg.inv(
            R.from_euler('xyz', curr_pos[3:6]).as_dcm()))).as_euler('xyz')))
        return cart_offset < cart_threshold and or_offset < or_threshold

    def reached_joint_state(self, joint_state):
        curr_pos = self.get_info()['joint_positions']
        offset = np.sum(np.abs((np.array(joint_state) - curr_pos)))
        return offset < 0.001

    def move_to_pose(self, pose, blocking=True):
        self.send_cartesian_coords_abs_PTP(pose)
        if blocking:
            while not self.reached_position(pose):
                time.sleep(0.05)


def work_position(controller):
    controller.send_cartesian_coords_abs_PTP((0, -0.56, 0.26, pi, 0, pi / 2))


def mechanical_zero(udp):
    udp.send_joint_angles((0, 0, 0, 0, 0, 0, 0))

def iiwa_error_debug(iiwa):
    for i in range(100):
        print(i)
        if i% 2 == 0:
            iiwa.send_cartesian_coords_abs_LIN((0, -0.556, 0.20, pi, 0, pi / 4))
        else:
            iiwa.send_cartesian_coords_abs_LIN((0, -0.556, 0.25, pi, 0, pi / 4))
        time.sleep(3)

if __name__ == "__main__":
    iiwa = IIWAController(use_impedance=True)
    iiwa._send_init_message()
    # work_position(iiwa)
    # print(iiwa.get_joint_info()[:6])
    # iiwa_error_debug(iiwa)
    # iiwa.send_joint_angles((-90, 30, 0, -90, 0, 60, 0))
    # mechanical_zero(iiwa)
    iiwa.send_cartesian_coords_abs_PTP((0, -0.5, 0.25, pi, 0, pi / 2))
    time.sleep(3)
    iiwa.send_cartesian_coords_rel_PTP((0,0,0.01,0,0,0))
    time.sleep(3)
    iiwa.send_cartesian_coords_abs_LIN((0, -0.5, 0.25, pi, 0, pi / 2))
    time.sleep(3)
    iiwa.send_cartesian_coords_rel_LIN((0, 0, 0.01, 0, 0, 0))
    # time.sleep(5)
    # iiwa.send_cartesian_coords_abs_LIN((0, -0.71, 0.3, pi, 0, pi / 2))
    # time.sleep(5)
    # iiwa.send_cartesian_coords_abs_LIN((0, -0.71, 0.25, pi, 0, pi / 2))
