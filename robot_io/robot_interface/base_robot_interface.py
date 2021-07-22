from enum import Enum
from scipy.spatial.transform.rotation import Rotation as R
import numpy as np


class GripperState(Enum):
    OPEN = 1
    CLOSED = -1


class BaseRobotInterface:
    def __init__(self, *args, **kwargs):
        pass

    def move_to_neutral(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_tcp_pose(self):
        raise NotImplementedError

    def get_tcp_pos_orn(self):
        raise NotImplementedError

    def move_async_cart_pos_abs_ptp(self, target_pos, target_orn):
        raise NotImplementedError

    def move_async_cart_pos_abs_lin(self, target_pos, target_orn):
        raise NotImplementedError

    def move_async_cart_pos_rel_ptp(self, rel_target_pos, rel_target_orn):
        raise NotImplementedError

    def move_async_cart_pos_rel_lin(self, rel_target_pos, rel_target_orn):
        raise NotImplementedError

    def move_async_joint_pos(self, joint_positions):
        raise NotImplementedError

    def close_gripper(self):
        raise NotImplementedError

    def open_gripper(self):
        raise NotImplementedError

    def reached_position(self, target_pos, target_orn, cart_threshold=0.005, orn_threshold=0.05):
        curr_pos, curr_orn = self.get_tcp_pos_orn()
        cart_offset = np.linalg.norm(target_pos - curr_pos)
        angle_diff = (R.from_euler('xyz', target_orn) * R.from_quat(curr_orn).inv()).as_euler('xyz')
        or_offset = np.sum(np.abs(angle_diff))
        return cart_offset < cart_threshold and or_offset < orn_threshold
