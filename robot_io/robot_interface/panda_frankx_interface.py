import time

import cv2
import numpy as np
from omegaconf import OmegaConf
import hydra.utils
import quaternion

from robot_io.robot_interface.base_robot_interface import BaseRobotInterface, GripperState
from frankx import Affine, JointMotion, LinearMotion, Robot, PathMotion, WaypointMotion, Waypoint, LinearRelativeMotion, StopMotion, ImpedanceMotion, JointWaypointMotion
from frankx.gripper import Gripper
from robot_io.utils.utils import np_quat_to_scipy_quat, pos_orn_to_matrix, quat_to_euler, euler_to_quat, \
    scipy_quat_to_np_quat, matrix_to_pos_orn
import logging
log = logging.getLogger(__name__)


def to_affine(target_pos, target_orn):
    if len(target_orn) == 3:
        target_orn = euler_to_quat(target_orn)
    target_orn = scipy_quat_to_np_quat(target_orn)
    return Affine(*target_pos, target_orn.w, target_orn.x, target_orn.y, target_orn.z)


class PandaFrankXInterface(BaseRobotInterface):
    def __init__(self,
                 fci_ip,
                 velocity_rel,
                 acceleration_rel,
                 jerk_rel,
                 neutral_pose,
                 ik,
                 workspace_limits,
                 use_impedance,
                 franka_joint_impedance,
                 translational_stiffness,
                 rotational_stiffness,
                 joint_stiffness,
                 gripper_speed,
                 gripper_force):
        self.name = "panda"
        self.neutral_pose = neutral_pose
        self.use_impedance = use_impedance
        self.joint_stiffness = joint_stiffness
        self.rotational_stiffness = rotational_stiffness
        self.translational_stiffness = translational_stiffness

        self.robot = Robot(fci_ip)
        self.robot.recover_from_errors()
        self.robot.set_default_behavior()
        self.robot.velocity_rel = velocity_rel
        self.robot.acceleration_rel = acceleration_rel
        self.robot.jerk_rel = jerk_rel
        self.robot.set_joint_impedance(franka_joint_impedance)
        self.gripper_thread = None
        self.motion_thread = None
        self.current_motion = None
        self.gripper = Gripper(fci_ip, gripper_speed, gripper_force)
        self.gripper_state = GripperState.CLOSED
        self.open_gripper(blocking=True)
        self.gripper_state = GripperState.OPEN

        self.ik_solver = hydra.utils.instantiate(ik)
        # FrankX needs continuous Euler angles around TCP, as the trajectory generation works in the Euler space.
        # internally, FrankX expects orientations with the z-axis facing up, but to be consistent with other
        # robot interfaces we transform the TCP orientation such that the z-axis faces down.
        self.NE_T_EE = self.EE_T_NE = Affine(0, 0, 0, 0, 0, np.pi)
        super().__init__()

    def __del__(self):
        if self.gripper_thread is not None:
            self.gripper_thread.join()
        self.abort_motion()

    def move_to_neutral(self):
        self.move_joint_pos(self.neutral_pose)

    def move_cart_pos_abs_ptp(self, target_pos, target_orn):
        if self.use_impedance:
            log.warning("Impedance motion for cartesian PTP is currently not implemented")
        q_desired = self._inverse_kinematics(target_pos, target_orn)
        self.move_joint_pos(q_desired)

    def move_async_cart_pos_abs_ptp(self, target_pos, target_orn):
        if self.use_impedance:
            log.warning("Impedance motion for cartesian PTP is currently not implemented")
        q_desired = self._inverse_kinematics(target_pos, target_orn)
        self.move_async_joint_pos(q_desired)

    def move_cart_pos_abs_lin(self, target_pos, target_orn):
        self.abort_motion()
        target_pose = to_affine(target_pos, target_orn) * self.NE_T_EE
        self.current_motion = WaypointMotion([Waypoint(target_pose)])
        self.robot.move(self.current_motion)

    def move_async_cart_pos_abs_lin(self, target_pos, target_orn):
        target_pose = to_affine(target_pos, target_orn) * self.NE_T_EE
        if self.use_impedance:
            # target_pose_rel = target_pose * (to_affine(*self.get_tcp_pos_orn()) * self.NE_T_EE).inverse()
            if self.current_motion is not None and isinstance(self.current_motion, ImpedanceMotion):
                self.current_motion.target = target_pose
            else:
                if self.current_motion is not None and not isinstance(self.current_motion, WaypointMotion):
                    self.abort_motion()
                self.current_motion = ImpedanceMotion(self.translational_stiffness, self.rotational_stiffness)
                self.current_motion.target = target_pose
                self.motion_thread = self.robot.move_async(self.current_motion)
        else:
            if self.current_motion is not None and isinstance(self.current_motion, WaypointMotion):
                self.current_motion.set_next_waypoint(Waypoint(target_pose))
            else:
                if self.current_motion is not None and not isinstance(self.current_motion, WaypointMotion):
                    self.abort_motion()
                self.current_motion = WaypointMotion([Waypoint(target_pose), ], return_when_finished=False)
                self.motion_thread = self.robot.move_async(self.current_motion)

    def move_async_joint_pos(self, joint_positions):
        if self.current_motion is not None and isinstance(self.current_motion, JointWaypointMotion):
            self.current_motion.set_next_target(joint_positions)
        else:
            if self.current_motion is not None and not isinstance(self.current_motion, JointWaypointMotion):
                self.abort_motion()
            self.current_motion = JointWaypointMotion([joint_positions], return_when_finished=False)
            self.motion_thread = self.robot.move_async(self.current_motion)

    def move_joint_pos(self, joint_positions):
        self.abort_motion()
        self.robot.move(JointMotion(joint_positions))

    def abort_motion(self):
        if self.current_motion is not None:
            self.current_motion.stop()
            self.current_motion = None
        if self.motion_thread is not None:
        #     # self.motion_thread.stop()
            self.motion_thread.join()
            self.motion_thread = None
        self.robot.recover_from_errors()

    def get_state(self):
        if self.current_motion is None:
            _state = self.robot.read_once()
        else:
            _state = self.current_motion.get_robot_state()
        pos, orn = self.get_tcp_pos_orn()
        state = {"tcp_pos": pos,
                 "tcp_orn": orn,
                 "joint_positions": np.array(_state.q),
                 "gripper_opening_width": self.gripper.width(),
                 "force_torque": np.array(_state.K_F_ext_hat_K)}
        return state

    def get_tcp_pos_orn(self):
        if self.current_motion is None:
            pose = self.robot.current_pose() * self.EE_T_NE
        else:
            pose = self.current_motion.current_pose() * self.EE_T_NE
        return np.array(pose.translation()), np.array(pose.quaternion())

    def get_tcp_pose(self):
        return pos_orn_to_matrix(*self.get_tcp_pos_orn())

    def open_gripper(self, blocking=False):
        if self.gripper_state == GripperState.CLOSED and (self.gripper_thread is None or (self.gripper_thread is not None and not self.gripper_thread.is_alive())):
            self.gripper_thread = self.gripper.move_async(0.085)
            if blocking:
                self.gripper_thread.join()
            self.gripper_state = GripperState.OPEN

    def close_gripper(self, blocking=False):
        if self.gripper_state == GripperState.OPEN and (self.gripper_thread is None or (self.gripper_thread is not None and not self.gripper_thread.is_alive())):
            self.gripper_thread = self.gripper.move_async_grasp(0)
            if blocking:
                self.gripper_thread.join()
            self.gripper_state = GripperState.CLOSED

    def _inverse_kinematics(self, target_pos, target_orn):
        """
        :param target_pos: cartesian target position
        :param target_orn: cartesian target orientation
        :return: target_joint_positions
        """
        current_q = self.get_state()['joint_positions']
        new_q = self.ik_solver.inverse_kinematics(target_pos, target_orn, current_q)
        return new_q

    def visualize_joint_states(self):
        canvas = np.ones((300, 300, 3))
        joint_states = self.get_state()["joint_positions"]
        left = 10
        right = 290
        width = right - left
        height = 30
        y = 10
        for i, (l, q, u) in enumerate(zip(self.ik_solver.ll, joint_states, self.ik_solver.ul)):
            cv2.rectangle(canvas, [left, y], [right, y + height], [0,0,0], thickness=2)
            bar_pos = int(left + width * (q - l) / (u - l))
            cv2.line(canvas, [bar_pos, y], [bar_pos, y + height], thickness=5, color=[0, 0, 1])
            y += height + 10
        cv2.imshow("joint_positions", canvas)
        cv2.waitKey(1)


@hydra.main(config_path="../conf", config_name="panda_teleop.yaml")
def main(cfg):
    robot = hydra.utils.instantiate(cfg.robot)
    robot.move_to_neutral()

    pos, orn = robot.get_tcp_pos_orn()

    for dy in (.02, -.02):
        new_pos = pos.copy()
        new_pos[1] += dy
        robot.move_cart_pos_abs_ptp(new_pos, orn)
        #robot.visualize_joint_states()
    robot.move_to_neutral()
    print("done!")

if __name__ == "__main__":
    main()
