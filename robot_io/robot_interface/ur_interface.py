import time

import hydra
import numpy as np

from robot_io.control.rel_action_control import RelActionControl
from robot_io.robot_interface.base_robot_interface import BaseRobotInterface
import rtde_control
import rtde_receive

from robot_io.utils.utils import ReferenceType, rotvec_to_quat, rotvec_to_euler, quat_to_rotvec, euler_to_rotvec, \
    timeit, pos_orn_to_matrix


def to_ur_pose(pos, orn):
    if len(orn) == 4:
        rotvec = quat_to_rotvec(orn)
    elif len(orn) == 3:
        rotvec = euler_to_rotvec(orn)
    else:
        raise ValueError
    return np.concatenate([pos, rotvec])


class URInterface(BaseRobotInterface):
    def __init__(self,
                 robot_ip,
                 gripper,
                 neutral_pose,
                 workspace_limits,
                 cartesian_speed,
                 cartesian_acc,
                 joint_speed,
                 joint_acc,
                 ll,
                 ul,
                 tcp_offset,
                 rel_action_params):
        self.name = "ur3"
        self.neutral_pose = neutral_pose
        self.reference_type = ReferenceType.ABSOLUTE
        self.cartesian_speed = cartesian_speed
        self.cartesian_acc = cartesian_acc
        self.joint_speed = joint_speed
        self.joint_acc = joint_acc

        try:
            self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        except RuntimeError as e:
            if e.args == ('ur_rtde: Failed to start control script, before timeout of 5 seconds',):
                print("ProTip: ur_rtde : maybe robot is not active")
            self.rtde_c = None
            self.rtde_r = None
            raise

        self.rtde_c.setTcp(tcp_offset)
        self.gripper = hydra.utils.instantiate(gripper)
        self.open_gripper(blocking=True)

        self.rel_action_converter = RelActionControl(ll=ll, ul=ul, workspace_limits=workspace_limits,
                                                     **rel_action_params)
        super().__init__(ll=ll, ul=ul)

    def __del__(self):
        if self.rtde_c is None:
            return
        self.abort_motion()
        self.rtde_c.stopScript()

    def move_to_neutral(self):
        return self.move_joint_pos(self.neutral_pose)

    def move_cart_pos_abs_ptp(self, target_pos, target_orn):
        self.reference_type = ReferenceType.ABSOLUTE
        self.abort_motion()
        pose = to_ur_pose(target_pos, target_orn)
        self.rtde_c.moveJ_IK(pose, self.joint_speed, self.joint_acc, False)

    def move_cart_pos_rel_ptp(self, rel_target_pos, rel_target_orn):
        target_pos, target_orn = self.rel_action_converter.to_absolute(rel_target_pos, rel_target_orn, self.get_state(), self.reference_type)
        self.reference_type = ReferenceType.RELATIVE
        self.abort_motion()
        pose = to_ur_pose(target_pos, target_orn)
        self.rtde_c.moveJ_IK(pose, self.joint_speed, self.joint_acc, False)

    def move_async_cart_pos_abs_ptp(self, target_pos, target_orn):
        self.reference_type = ReferenceType.ABSOLUTE
        pose = to_ur_pose(target_pos, target_orn)
        self.rtde_c.moveJ_IK(pose, self.joint_speed, self.joint_acc, asynchronous=True)

    def move_async_cart_pos_rel_ptp(self, rel_target_pos, rel_target_orn):
        target_pos, target_orn = self.rel_action_converter.to_absolute(rel_target_pos, rel_target_orn, self.get_state(), self.reference_type)
        self.reference_type = ReferenceType.RELATIVE
        pose = to_ur_pose(target_pos, target_orn)
        self.rtde_c.moveJ_IK(pose, self.joint_speed, self.joint_acc, asynchronous=True)

    def move_cart_pos_abs_lin(self, target_pos, target_orn):
        self.reference_type = ReferenceType.ABSOLUTE
        self.abort_motion()
        pose = to_ur_pose(target_pos, target_orn)
        self.rtde_c.moveL(pose, self.cartesian_speed, self.cartesian_acc, False)

    def move_cart_pos_rel_lin(self, rel_target_pos, rel_target_orn):
        target_pos, target_orn = self.rel_action_converter.to_absolute(rel_target_pos, rel_target_orn, self.get_state(), self.reference_type)
        self.reference_type = ReferenceType.RELATIVE
        self.abort_motion()
        pose = to_ur_pose(target_pos, target_orn)
        self.rtde_c.moveL(pose, self.cartesian_speed, self.cartesian_acc, False)

    def move_async_cart_pos_abs_lin(self, target_pos, target_orn):
        self.reference_type = ReferenceType.ABSOLUTE
        pose = to_ur_pose(target_pos, target_orn)
        self.rtde_c.moveL(pose, self.cartesian_speed, self.cartesian_acc, asynchronous=True)

    def move_async_cart_pos_rel_lin(self, rel_target_pos, rel_target_orn):
        target_pos, target_orn = self.rel_action_converter.to_absolute(rel_target_pos, rel_target_orn, self.get_state(), self.reference_type)
        self.reference_type = ReferenceType.RELATIVE
        pose = to_ur_pose(target_pos, target_orn)
        # self.rtde_c.moveL(pose, self.cartesian_speed, self.cartesian_acc, asynchronous=False)
        self.rtde_c.servoL(pose, self.cartesian_speed, self.cartesian_acc, 0.05, 0.2, 100)

    def abort_motion(self):
        if self.reference_type == ReferenceType.JOINT:
            self.rtde_c.stopJ(0.5)
        else:
            self.rtde_c.stopL(0.5)

    def move_joint_pos(self, joint_positions):
        self.reference_type = ReferenceType.JOINT
        self.abort_motion()
        return self.rtde_c.moveJ(joint_positions, self.joint_speed, self.joint_acc, False)

    def get_state(self):
        pos, orn = self.get_tcp_pos_orn()
        state = {"tcp_pos": pos,
                 "tcp_orn": orn,
                 "joint_positions": np.array(self.rtde_r.getActualQ()),
                 "gripper_opening_width": None,
                 "force_torque": np.array(self.rtde_r.getActualTCPForce()),
                 "contact": np.zeros(6)}
        return state

    def get_tcp_pos_orn(self):
        pose = np.array(self.rtde_r.getActualTCPPose())
        pos, orn = pose[:3], rotvec_to_quat(pose[3:])
        return pos, orn

    def get_tcp_pose(self):
        return pos_orn_to_matrix(*self.get_tcp_pos_orn())

    def open_gripper(self, blocking=False):
        self.gripper.open_gripper(blocking)

    def close_gripper(self, blocking=False):
        self.gripper.close_gripper(blocking)

    def visualize_external_forces(self, canvas_width=500):
        """
        Display the external forces (x,y,z) and torques (a,b,c) of the tcp frame.

        Args:
            canvas_width: Display width in pixel.

        """
        contact = np.array([50, 50, 50, 50, 50, 50])
        collision = contact
        self._visualize_external_forces(contact, collision, canvas_width)


if __name__ == "__main__":
    hydra.initialize("../conf/robot/")
    cfg = hydra.compose("ur_interface.yaml")
    robot = hydra.utils.instantiate(cfg)

    robot.move_to_neutral()
    pos = np.array([0, 0.2, 0])
    orn = np.array([0, 0, 0])
    robot.move_async_cart_pos_rel_lin(pos, orn)
    print(robot.get_state())
    time.sleep(1)
    print(robot.get_state())
    robot.close_gripper()
    time.sleep(1)
