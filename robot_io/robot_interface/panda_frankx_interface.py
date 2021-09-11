import time
from enum import Enum

import cv2
import numpy as np
from omegaconf import OmegaConf
import hydra.utils
import quaternion

from robot_io.robot_interface.base_robot_interface import BaseRobotInterface, GripperState
from frankx import Affine, JointMotion, LinearMotion, Robot, PathMotion, WaypointMotion, Waypoint, LinearRelativeMotion, StopMotion, ImpedanceMotion, JointWaypointMotion
from frankx.gripper import Gripper
from robot_io.utils.utils import np_quat_to_scipy_quat, pos_orn_to_matrix, quat_to_euler, euler_to_quat, \
    scipy_quat_to_np_quat, orn_to_matrix, matrix_to_orn, timeit
import logging
log = logging.getLogger(__name__)

DEFAULT_ROT_X = np.pi
DEFAULT_ROT_Y = 0


def to_affine(target_pos, target_orn):
    if len(target_orn) == 3:
        target_orn = euler_to_quat(target_orn)
    target_orn = scipy_quat_to_np_quat(target_orn)
    return Affine(*target_pos, target_orn.w, target_orn.x, target_orn.y, target_orn.z)


class ReferenceType(Enum):
    ABSOLUTE = 0
    RELATIVE = 1
    JOINT = 2


class PandaFrankXInterface(BaseRobotInterface):
    def __init__(self,
                 fci_ip,
                 velocity_rel,
                 acceleration_rel,
                 jerk_rel,
                 neutral_pose,
                 ik,
                 workspace_limits,
                 ll,
                 ul,
                 contact_torque_threshold,
                 collision_torque_threshold,
                 contact_force_threshold,
                 collision_force_threshold,
                 use_impedance,
                 franka_joint_impedance,
                 translational_stiffness,
                 rotational_stiffness,
                 joint_stiffness,
                 gripper_speed,
                 gripper_force,
                 gripper_timeout,
                 gripper_opening_threshold,
                 gripper_closing_threshold,
                 relative_action_reference_frame,
                 relative_action_control_frame,
                 relative_pos_clip_threshold,
                 relative_rot_clip_threshold,
                 limit_control_5_dof):
        self.name = "panda"
        self.neutral_pose = neutral_pose
        self.use_impedance = use_impedance
        self.joint_stiffness = joint_stiffness
        self.rotational_stiffness = rotational_stiffness
        self.translational_stiffness = translational_stiffness
        self.ll = np.array(ll)
        self.ul = np.array(ul)
        self.workspace_limits = workspace_limits
        assert relative_action_reference_frame in ["current", "desired"]
        self.relative_action_reference_frame = relative_action_reference_frame
        assert relative_action_control_frame in ["world", "tcp"]
        self.relative_action_control_frame = relative_action_control_frame
        self.relative_pos_clip_threshold = relative_pos_clip_threshold
        self.relative_rot_clip_threshold = relative_rot_clip_threshold
        self.limit_control_5_dof = limit_control_5_dof
        self.desired_pos, self.desired_orn = None, None
        self.robot = Robot(fci_ip)
        self.robot.recover_from_errors()
        self.robot.set_default_behavior()
        self.set_collision_behavior(contact_torque_threshold,
                                    collision_torque_threshold,
                                    contact_force_threshold,
                                    collision_force_threshold)
        self.robot.velocity_rel = velocity_rel
        self.robot.acceleration_rel = acceleration_rel
        self.robot.jerk_rel = jerk_rel
        self.robot.set_joint_impedance(franka_joint_impedance)
        self.gripper_thread = None
        self.motion_thread = None
        self.current_motion = None
        self.gripper = Gripper(fci_ip, gripper_speed, gripper_force, gripper_timeout, gripper_opening_threshold, gripper_closing_threshold)
        self.open_gripper(blocking=True)
        F_T_NE = np.array(self.robot.read_once().F_T_NE).reshape((4, 4)).T
        self.ik_solver = hydra.utils.instantiate(ik, F_T_NE=F_T_NE)
        # FrankX needs continuous Euler angles around TCP, as the trajectory generation works in the Euler space.
        # internally, FrankX expects orientations with the z-axis facing up, but to be consistent with other
        # robot interfaces we transform the TCP orientation such that the z-axis faces down.
        self.NE_T_EE = self.EE_T_NE = Affine(0, 0, 0, 0, 0, np.pi)
        self.reference_type = ReferenceType.ABSOLUTE
        super().__init__()

    def __del__(self):
        if self.gripper_thread is not None:
            self.gripper_thread.join()
        self.abort_motion()

    def set_collision_behavior(self,
                               contact_torque_threshold,
                               collision_torque_threshold,
                               contact_force_threshold,
                               collision_force_threshold):
        self.robot.set_collision_behavior([contact_torque_threshold] * 7,
                                          [collision_torque_threshold] * 7,
                                          [contact_force_threshold] * 6,
                                          [collision_force_threshold] * 6)


    def move_to_neutral(self):
        self.move_joint_pos(self.neutral_pose)

    def move_cart_pos_abs_ptp(self, target_pos, target_orn):
        self.reference_type = ReferenceType.ABSOLUTE
        if self.use_impedance:
            log.warning("Impedance motion is not available for synchronous motions. Not using impedance.")
        q_desired = self._inverse_kinematics(target_pos, target_orn)
        self.move_joint_pos(q_desired)

    def move_async_cart_pos_rel_lin(self, rel_target_pos, rel_target_orn):
        target_pos, target_orn = self._relative_to_absolute(rel_target_pos, rel_target_orn)
        self.reference_type = ReferenceType.RELATIVE
        self._frankx_async_impedance_motion(target_pos, target_orn)

    def move_async_cart_pos_abs_ptp(self, target_pos, target_orn):
        self.reference_type = ReferenceType.ABSOLUTE
        if self.use_impedance:
            log.warning("Impedance motion for cartesian PTP is currently not implemented. Not using impedance.")
        q_desired = self._inverse_kinematics(target_pos, target_orn)
        self.move_async_joint_pos(q_desired)

    def move_cart_pos_abs_lin(self, target_pos, target_orn):
        self.reference_type = ReferenceType.ABSOLUTE
        if self.use_impedance:
            log.warning("Impedance motion for cartesian LIN is currently not implemented. Not using impedance.")
        self.abort_motion()
        target_pose = to_affine(target_pos, target_orn) * self.NE_T_EE
        self.current_motion = WaypointMotion([Waypoint(target_pose)])
        self.robot.move(self.current_motion)

    def move_async_cart_pos_abs_lin(self, target_pos, target_orn):
        self.reference_type = ReferenceType.ABSOLUTE
        if self.use_impedance:
            self._frankx_async_impedance_motion(target_pos, target_orn)
        else:
            self._frankx_async_lin_motion(target_pos, target_orn)

    def move_async_joint_pos(self, joint_positions):
        if self.current_motion is not None and isinstance(self.current_motion, JointWaypointMotion):
            self.current_motion.set_next_target(joint_positions)
        else:
            if self.current_motion is not None and not isinstance(self.current_motion, JointWaypointMotion):
                self.abort_motion()
            self.current_motion = JointWaypointMotion([joint_positions], return_when_finished=False)
            self.motion_thread = self.robot.move_async(self.current_motion)

    def move_joint_pos(self, joint_positions):
        self.reference_type = ReferenceType.JOINT
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

    def _relative_to_absolute(self, rel_target_pos, rel_target_orn):
        state = self.get_state()
        tcp_pos, tcp_orn, joint_positions = state["tcp_pos"], state["tcp_orn"], state["joint_positions"]
        if self.limit_control_5_dof:
            rel_target_orn = self._enforce_limit_gripper_joint(joint_positions, rel_target_orn)
        rel_target_pos = self._restrict_action_if_contact(rel_target_pos, state)
        if self.relative_action_control_frame == "tcp":
            rel_target_pos = orn_to_matrix(tcp_orn) @ self.NE_T_EE.rotation() @ rel_target_pos
        if self.relative_action_reference_frame == "current":
            tcp_orn = quat_to_euler(tcp_orn)
            abs_target_pos = tcp_pos + rel_target_pos
            abs_target_orn = tcp_orn + rel_target_orn
            if self.limit_control_5_dof:
                abs_target_orn = self._enforce_5_dof_control(abs_target_orn)
            abs_target_pos = self._restrict_workspace(abs_target_pos)
            return abs_target_pos, abs_target_orn
        else:
            if self.reference_type != ReferenceType.RELATIVE:
                self.desired_pos, self.desired_orn = tcp_pos, tcp_orn
                self.desired_orn = quat_to_euler(self.desired_orn)
                self.desired_orn = self._enforce_5_dof_control(self.desired_orn)
            self.desired_pos, self.desired_orn = self._clip_relative_action(rel_target_pos, rel_target_orn, tcp_pos, tcp_orn)
            self.desired_pos = self._restrict_workspace(self.desired_pos)
            return self.desired_pos, self.desired_orn

    @staticmethod
    def _enforce_5_dof_control(abs_target_orn):
        assert len(abs_target_orn) == 3
        abs_target_orn[0] = DEFAULT_ROT_X
        abs_target_orn[1] = DEFAULT_ROT_Y
        return abs_target_orn

    def _enforce_limit_gripper_joint(self, joint_positions, rel_target_orn):
        if rel_target_orn[2] < 0 and joint_positions[-1] - rel_target_orn[2] > self.ul[-1] * 0.8:
            rel_target_orn[2] = 0
        elif rel_target_orn[2] > 0 and joint_positions[-1] - rel_target_orn[2] < self.ll[-1] * 0.8:
            rel_target_orn[2] = 0
        return rel_target_orn

    @staticmethod
    def _restrict_action_if_contact(rel_target_pos, state):
        if state["contact"][2] and rel_target_pos[2] < 0:
            print("contact!")
            rel_target_pos[2] *= -1
        return rel_target_pos

    def _clip_relative_action(self, rel_target_pos, rel_target_orn, tcp_pos, tcp_orn):
        # limit position
        for i in range(3):
            if rel_target_pos[i] > 0 and self.desired_pos[i] - tcp_pos[i] < self.relative_pos_clip_threshold:
                self.desired_pos[i] += rel_target_pos[i]
            elif rel_target_pos[i] < 0 and tcp_pos[i] - self.desired_pos[i] < self.relative_pos_clip_threshold:
                self.desired_pos[i] += rel_target_pos[i]
        # limit orientation
        rot_diff = quat_to_euler(matrix_to_orn(np.linalg.inv(orn_to_matrix(self.desired_orn)) @ orn_to_matrix(tcp_orn)))
        for i in range(3):
            if rel_target_orn[i] > 0 and rot_diff[i] < self.relative_rot_clip_threshold:
                self.desired_orn[i] += rel_target_orn[i]
            elif rel_target_orn[i] < 0 and rot_diff[i] > -self.relative_rot_clip_threshold:
                self.desired_orn[i] += rel_target_orn[i]
        return self.desired_pos, self.desired_orn

    def _frankx_async_impedance_motion(self, target_pos, target_orn):
        target_pose = to_affine(target_pos, target_orn) * self.NE_T_EE
        if self.current_motion is not None and isinstance(self.current_motion, ImpedanceMotion):
            self.current_motion.target = target_pose
        else:
            if self.current_motion is not None and not isinstance(self.current_motion, WaypointMotion):
                self.abort_motion()
            self.current_motion = ImpedanceMotion(self.translational_stiffness, self.rotational_stiffness)
            self.current_motion.target = target_pose
            self.motion_thread = self.robot.move_async(self.current_motion)

    def _frankx_async_lin_motion(self, target_pos, target_orn):
        target_pose = to_affine(target_pos, target_orn) * self.NE_T_EE
        if self.current_motion is not None and isinstance(self.current_motion, WaypointMotion):
            self.current_motion.set_next_waypoint(Waypoint(target_pose))
        else:
            if self.current_motion is not None and not isinstance(self.current_motion, WaypointMotion):
                self.abort_motion()
            self.current_motion = WaypointMotion([Waypoint(target_pose), ], return_when_finished=False)
            self.motion_thread = self.robot.move_async(self.current_motion)

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
                 "force_torque": np.array(_state.K_F_ext_hat_K),
                 "contact": _state.cartesian_contact}
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
        self.gripper.open(blocking)

    def close_gripper(self, blocking=False):
        self.gripper.close(blocking)

    def _inverse_kinematics(self, target_pos, target_orn):
        """
        :param target_pos: cartesian target position
        :param target_orn: cartesian target orientation
        :return: target_joint_positions
        """
        current_q = self.get_state()['joint_positions']
        new_q = self.ik_solver.inverse_kinematics(target_pos, target_orn, current_q)
        return new_q

    def _restrict_workspace(self, target_pos):
        """
        :param target_pos: cartesian target position
        :return: clip target_pos at workspace limits
        """
        return np.clip(target_pos, self.workspace_limits[0], self.workspace_limits[1])

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

    # pos[2] -= 0.05
    print("move")
    robot.move_cart_pos_abs_ptp(pos, orn)
    time.sleep(1)
    print("done!")


if __name__ == "__main__":
    main()
