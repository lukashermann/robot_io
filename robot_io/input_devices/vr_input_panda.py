import math

import time

import sys

import logging

import numpy as np
import pybullet as p
# import quaternion  # noqa
from numba.np.arraymath import np_all
from scipy.spatial.transform.rotation import Rotation as R
import pybullet_utils.bullet_client as bc

# A logger for this file
log = logging.getLogger(__name__)

GRIPPER_CLOSING_ACTION = -1
GRIPPER_OPENING_ACTION = 1


def z_angle_between(a, b):
    """
    :param a: 3d vector
    :param b: 3d vector
    :return: signed angle between vectors around z axis (right handed rule)
    """
    return math.atan2(b[1], b[0]) - math.atan2(a[1], a[0])


def scipy_quat_to_np_quat(quat):
    """xyzw to wxyz"""
    return np.quaternion(quat[3], quat[0], quat[1], quat[2])


def np_quat_to_scipy_quat(quat):
    """wxyz to xyzw"""
    return np.array([quat.x, quat.y, quat.z, quat.w])


def pos_orn_to_matrix(pos, orn):
    """
    :param pos: np.array of shape (3,)
    :param orn: np.array of shape (4,) -> quaternion xyzw
                np.quaternion -> quaternion wxyz
                np.array of shape (3,) -> euler angles xyz
    :return: 4x4 homogeneous transformation
    """
    mat = np.eye(4)
    if isinstance(orn, np.quaternion):
        orn = np_quat_to_scipy_quat(orn)
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 4:
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 3:
        mat[:3, :3] = R.from_euler('xyz', orn).as_matrix()
    mat[:3, 3] = pos
    return mat


def matrix_to_pos_orn(mat):
    """
    :param mat: 4x4 homogeneous transformation
    :return: tuple(position: np.array of shape (3,), orientation: np.array of shape (4,) -> quaternion xyzw)
    """
    orn = scipy_quat_to_np_quat(R.from_matrix(mat[:3, :3]).as_quat())
    pos = mat[:3, 3]
    return pos, orn


class VrInput:
    """
    This class processes the input of the vr controller for teleoperating a real franka emika panda robot.
    """

    def __init__(self, robot, quaternion_convention='wxyz'):
        """
        :param robot: instance of PandaArm class from panda_robot repository
        :param quaternion_convention: default output
        """
        self.robot = robot
        self.vr_controller_id = 3
        self.POSITION = 1
        self.ORIENTATION = 2
        self.ANALOG = 3
        self.BUTTONS = 6
        self.BUTTON_A = 2
        self.BUTTON_B = 1
        self.gripper_orientation_offset = R.from_euler('xyz', [0, 0, np.pi / 2])
        self.vr_pos_uid = None
        self.vr_coord_rotation = np.eye(4)
        self.change_quaternion_convention = False
        assert quaternion_convention in ('wxyz', 'xyzw')
        if quaternion_convention == 'wxyz':
            self.change_quaternion_convention = True
        self.p = None
        self.initialize_bullet()

        self.prev_action = None
        self.robot_start_pos_offset = None

    def initialize_bullet(self):
        self.p = bc.BulletClient(connection_mode=p.SHARED_MEMORY)
        cid = self.p._client
        if cid < 0:
            log.error("Failed to connect to SHARED_MEMORY bullet server.\n" " Is it running?")
            sys.exit(1)
        # self.p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=cid)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_VR_PICKING, 0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_VR_RENDER_CONTROLLERS, 0)
        print(f"Connected to server with id: {cid}")

    def get_vr_action(self):
        """
        :return: EE target pos, orn and gripper action  (in robot base frame)
        """
        vr_events = self.p.getVREvents()
        if vr_events != ():
            for event in vr_events:
                # if event[0] == self.vr_controller_id:
                vr_action = self.vr_event_to_action(event)
                # if "dead man's switch" is not pressed, do not update pose
                if not (event[self.BUTTONS][self.BUTTON_A] & p.VR_BUTTON_IS_DOWN):
                    return self.prev_action
                # reset button pressed
                elif event[self.BUTTONS][self.BUTTON_A] & p.VR_BUTTON_WAS_TRIGGERED:
                    self.reset(vr_action)

                # transform pose from vr coord system to robot base frame
                robot_action = self.transform_action_vr_to_robot_base(vr_action)
                self.prev_action = robot_action
        return self.prev_action

    def reset(self, vr_action):
        """
        This is called when the dead man's switch is triggered. The current vr controller pose is
        taken as origin for the proceeding vr motion.
        :param vr_action: the current vr controller position, orientation and gripper action
        """
        print("reset")
        pos, orn = self.robot.ee_pose()
        T_VR = self.vr_coord_rotation @ pos_orn_to_matrix(vr_action[0], vr_action[1])

        self.robot_start_pos_offset = pos - T_VR[:3, 3]
        self.robot_start_orn_offset = R.from_matrix(T_VR[:3, :3]).inv() * R.from_quat(np_quat_to_scipy_quat(orn))
        print(self.robot_start_orn_offset.as_euler('xyz'))

    def transform_action_vr_to_robot_base(self, vr_action):
        """
        Transform the vr controller pose to the coordinate system of the robot base.
        Consider the vr pose
        :param vr_action: vr_pos, vr_orn, gripper action (in vr coord system)
        :return: robot_pos, robot_orn, grip (in robot base frame)
        """
        vr_pos, vr_orn, grip = vr_action
        # rotate vr coord system to align orientation with robot base frame
        T_VR_Controller = self.vr_coord_rotation @ pos_orn_to_matrix(vr_action[0], vr_action[1])
        # robot pos and orn are calculated relative to last reset
        robot_pos = T_VR_Controller[:3, 3] + self.robot_start_pos_offset
        robot_orn = R.from_matrix(T_VR_Controller[:3, :3]) * self.robot_start_orn_offset
        robot_orn = scipy_quat_to_np_quat(robot_orn.as_quat())
        return robot_pos, robot_orn, grip

    def vr_event_to_action(self, event):
        """
        :param event: pybullet VR event
        :return: vr_controller_pos, vr_controller_orn, gripper_action
        """
        vr_controller_pos = np.array(event[self.POSITION])
        vr_controller_orn = np.array(event[self.ORIENTATION])
        controller_analogue_axis = event[self.ANALOG]

        gripper_action = GRIPPER_CLOSING_ACTION if controller_analogue_axis > 0.1 else GRIPPER_OPENING_ACTION

        return vr_controller_pos, vr_controller_orn, gripper_action

    def wait_for_start_button(self):
        """
        Wait until dead man's switch is pressed once.
        """
        print("wait for start button press")
        action = None
        while action is None:
            action = self.get_vr_action()
            time.sleep(0.1)
        print("start button pressed")

    def calibrate_vr_coord_system(self):
        """
        Align the orientation of the vr coordinate system by moving the vr controller in x-direction.
        """
        start_pose = None
        end_pose = None
        print("Hold VR controller and press dead man's switch once")
        while True:
            vr_events = self.p.getVREvents()
            if vr_events != ():
                for event in vr_events:
                    # if event[0] == self.vr_controller_id:
                    vr_action = self.vr_event_to_action(event)
                    if event[self.BUTTONS][self.BUTTON_A] and p.VR_BUTTON_WAS_TRIGGERED and start_pose is None:
                        print("start pose set")
                        print("Now move vr controller in your preferred X-direction")
                        start_pose = pos_orn_to_matrix(vr_action[0], vr_action[1])
                    elif event[self.BUTTONS][
                        self.BUTTON_B] and p.VR_BUTTON_WAS_TRIGGERED and start_pose is not None and end_pose is None:
                        print("end pose set")
                        end_pose = pos_orn_to_matrix(vr_action[0], vr_action[1])

                    if start_pose is not None and end_pose is not None:
                        self.set_vr_coord_transformation(start_pose, end_pose)
                        return

    def set_vr_coord_transformation(self, start_pose, end_pose):
        """
        Calculate rotation between default VR coordinate system and user defined vr coordinate system.
        The x-axis of the user defined coordinate system is defined as the vector from start_pose to end_pose.
        :param start_pose: start of new x-axis
        :param end_pose: end of new x-axis
        """
        new_x = end_pose[:2, 3] - start_pose[:2, 3]
        new_x = new_x / np.linalg.norm(new_x)
        old_x = np.array([1, 0])
        z_angle = z_angle_between(new_x, old_x)
        self.vr_coord_rotation = np.eye(4)
        self.vr_coord_rotation[:3, :3] = R.from_euler('z', [z_angle]).as_matrix()
