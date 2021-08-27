

import time

import sys

import logging

import numpy as np
import pybullet as p
import quaternion
from numba.np.arraymath import np_all
from scipy.spatial.transform.rotation import Rotation as R
import pybullet_utils.bullet_client as bc
from robot_io.utils.utils import *

# A logger for this file
log = logging.getLogger(__name__)

GRIPPER_CLOSING_ACTION = -1
GRIPPER_OPENING_ACTION = 1

DEFAULT_RECORD_INFO = {"hold": False,
                       "hold_event": False,
                       "down": False,
                       "dead_man_switch_triggered": False,
                       "triggered": False,
                       "trigger_release": False}


class VrInput:
    """
    This class processes the input of the vr controller for teleoperating a real franka emika panda robot.
    """

    def __init__(self, robot, workspace_limits, quaternion_convention='wxyz', record_button_queue_len=60):
        """
        :param workspace_limits: workspace bounding box [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        :param robot: instance of PandaArm class from panda_robot repository
        :param quaternion_convention: default output
        :param record_button_queue_len: after how many steps a button counts as "hold"
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
        self.workspace_limits = workspace_limits
        self.change_quaternion_convention = False
        assert quaternion_convention in ('wxyz', 'xyzw')
        if quaternion_convention == 'wxyz':
            self.change_quaternion_convention = True
        self.p = None
        self.record_button_press_counter = 0
        self.record_button_queue_len = record_button_queue_len
        self._initialize_bullet()

        self.prev_action = None
        self.robot_start_pos_offset = None
        self.out_of_workspace_offset = np.zeros(3)
        self.prev_record_info = DEFAULT_RECORD_INFO

        self.calibrate_vr_coord_system()

    def _initialize_bullet(self):
        log.info("Trying to connect to bullet SHARED_MEMORY. Make sure bullet_vr is running.")
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

    def get_action(self):
        """
        :return: EE target pos, orn and gripper action  (in robot base frame)
        """
        record_info = DEFAULT_RECORD_INFO
        vr_events = self.p.getVREvents()
        if vr_events != ():
            assert len(vr_events) == 1, "Only one VR controller should be turned on at the same time."
            for event in vr_events:
                # if event[0] == self.vr_controller_id:
                vr_action = self._vr_event_to_action(event)

                record_info = self._get_record_info(event)

                # if "dead man's switch" is not pressed, do not update pose
                if not self._dead_mans_switch_down(event):
                    return self.prev_action, record_info
                # reset button pressed
                elif self._dead_mans_switch_triggered(event):
                    self._reset_vr_coord_offset(vr_action)

                # transform pose from vr coord system to robot base frame
                robot_action = self._transform_action_vr_to_robot_base(vr_action)
                robot_action = {"motion": robot_action, "ref": "abs"}
                self.prev_action = robot_action
        return self.prev_action, record_info

    def _get_record_info(self, event):
        record_button_hold = False
        if self._record_button_down(event):
            self.record_button_press_counter += 1
        else:
            self.record_button_press_counter = 0
        if self.record_button_press_counter >= self.record_button_queue_len:
            record_button_hold = True

        self.prev_record_info = {"hold_event": record_button_hold and not self.prev_record_info["hold"],
                                 "hold": record_button_hold,
                                 "down": self._record_button_down(event),
                                 "dead_man_switch_triggered": self._dead_mans_switch_triggered(event),
                                 "triggered": self._record_button_triggered(event) and self._record_button_down(event),
                                 "trigger_release": self._record_button_released(event) and not self.prev_record_info["hold"] and self.prev_record_info["down"]}
        return self.prev_record_info

    def _dead_mans_switch_down(self, event):
        return bool(event[self.BUTTONS][self.BUTTON_A] & p.VR_BUTTON_IS_DOWN)

    def _dead_mans_switch_triggered(self, event):
        return bool(event[self.BUTTONS][self.BUTTON_A] & p.VR_BUTTON_WAS_TRIGGERED)

    def _record_button_down(self, event):
        return bool(event[self.BUTTONS][self.BUTTON_B] & p.VR_BUTTON_IS_DOWN)

    def _record_button_triggered(self, event):
        return bool(event[self.BUTTONS][self.BUTTON_B] & p.VR_BUTTON_WAS_TRIGGERED)

    def _record_button_released(self, event):
        return bool(event[self.BUTTONS][self.BUTTON_B] & p.VR_BUTTON_WAS_RELEASED)

    def _reset_vr_coord_offset(self, vr_action):
        """
        This is called when the dead man's switch is triggered. The current vr controller pose is
        taken as origin for the proceeding vr motion.
        :param vr_action: the current vr controller position, orientation and gripper action
        """
        print("reset")
        pos, orn = self.robot.get_tcp_pos_orn()
        T_VR = self.vr_coord_rotation @ pos_orn_to_matrix(vr_action[0], vr_action[1])

        self.robot_start_pos_offset = pos - T_VR[:3, 3]
        self.robot_start_orn_offset = R.from_matrix(T_VR[:3, :3]).inv() * R.from_quat(orn)
        print(self.robot_start_orn_offset.as_euler('xyz'))

        self.out_of_workspace_offset = np.zeros(3)

    def _transform_action_vr_to_robot_base(self, vr_action):
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
        robot_orn = robot_orn.as_quat()

        robot_pos = self._enforce_workspace_limits(robot_pos)

        return robot_pos, robot_orn, grip

    def _enforce_workspace_limits(self, robot_pos):
        robot_pos -= self.out_of_workspace_offset
        self.out_of_workspace_offset += self.get_out_of_workspace_offset(robot_pos)
        robot_pos = np.clip(robot_pos, self.workspace_limits[0], self.workspace_limits[1])
        return robot_pos

    def get_out_of_workspace_offset(self, pos):
        return np.clip(pos - self.workspace_limits[0], [-np.inf] * 3, 0) + np.clip(pos - self.workspace_limits[1], 0,  [np.inf] * 3)

    def _vr_event_to_action(self, event):
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
            action = self.get_action()
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
                    vr_action = self._vr_event_to_action(event)
                    if self._dead_mans_switch_down(event) and start_pose is None:
                        print("start pose set")
                        print("Now move vr controller in your preferred X-direction")
                        start_pose = pos_orn_to_matrix(vr_action[0], vr_action[1])
                    elif self._record_button_down(event) and start_pose is not None and end_pose is None:
                        print("end pose set")
                        print("Now press the dead man's switch once before pressing record")
                        end_pose = pos_orn_to_matrix(vr_action[0], vr_action[1])

                    if start_pose is not None and end_pose is not None:
                        self._set_vr_coord_transformation(start_pose, end_pose)
                        return

    def _set_vr_coord_transformation(self, start_pose, end_pose):
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


if __name__ == "__main__":
    vr_input = VrInput(robot=None)
    print("sleep 3")
    time.sleep(3)
    print("enter loop")
    while True:
        action, info = vr_input.get_action()
        if info["triggered"]:
            print("triggered")
        if info["hold_event"]:
            print("hold_event")
        if info["trigger_release"]:
            print("trigger_release")
        time.sleep(0.01)