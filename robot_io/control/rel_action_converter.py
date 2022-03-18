import numpy as np

from robot_io.utils.utils import quat_to_euler, orn_to_matrix, matrix_to_orn, angle_between, to_world_frame, \
    to_tcp_frame, ReferenceType


class RelActionConverter:
    def __init__(self,
                 workspace_limits,
                 relative_action_reference_frame,
                 relative_action_control_frame,
                 relative_pos_clip_threshold,
                 relative_rot_clip_threshold,
                 limit_control_5_dof,
                 max_orn_x,
                 max_orn_z,
                 ll,
                 ul,
                 default_orn_x,
                 default_orn_y):

        self.workspace_limits = workspace_limits
        assert relative_action_reference_frame in ["current", "desired"]
        self.relative_action_reference_frame = relative_action_reference_frame
        assert relative_action_control_frame in ["world", "tcp"]
        self.relative_action_control_frame = relative_action_control_frame
        self.relative_pos_clip_threshold = relative_pos_clip_threshold
        self.relative_rot_clip_threshold = relative_rot_clip_threshold
        self.limit_control_5_dof = limit_control_5_dof
        self.max_orn_x = np.radians(max_orn_x)
        self.max_orn_z = np.radians(max_orn_z)
        self.ll = np.array(ll)
        self.ul = np.array(ul)
        self.default_orn_x = default_orn_x
        self.default_orn_y = default_orn_y
        self.desired_pos, self.desired_orn = None, None

    def to_absolute(self, rel_action_pos, rel_action_orn, state, reference_type):
        tcp_pos, tcp_orn, joint_positions = state["tcp_pos"], state["tcp_orn"], state["joint_positions"]
        if self.limit_control_5_dof:
            rel_action_orn = self._enforce_limit_gripper_joint(joint_positions, rel_action_orn)
        rel_action_pos, rel_action_orn = self._restrict_action_if_contact(rel_action_pos, rel_action_orn, state)

        if self.relative_action_reference_frame == "current":
            if self.relative_action_control_frame == "tcp":
                rel_action_pos = orn_to_matrix(tcp_orn) @ rel_action_pos
            tcp_orn = quat_to_euler(tcp_orn)
            abs_target_pos = tcp_pos + rel_action_pos
            abs_target_orn = tcp_orn + rel_action_orn
            if self.limit_control_5_dof:
                abs_target_orn = self._enforce_5_dof_control(abs_target_orn)
            abs_target_pos = self._restrict_workspace(abs_target_pos)
            return abs_target_pos, abs_target_orn
        else:
            if reference_type != ReferenceType.RELATIVE:
                self.desired_pos, self.desired_orn = tcp_pos, tcp_orn
                self.desired_orn = quat_to_euler(self.desired_orn)
                if self.limit_control_5_dof:
                    self.desired_orn = self._enforce_5_dof_control(self.desired_orn)
            if self.relative_action_control_frame == "tcp":
                # rel_action_pos = orn_to_matrix(self.desired_orn) @ rel_action_pos
                rel_action_pos, rel_action_orn = to_world_frame(rel_action_pos, rel_action_orn, self.desired_orn)
            self.desired_pos, desired_orn = self._apply_relative_action(rel_action_pos, rel_action_orn, tcp_pos, tcp_orn)
            self.desired_orn = self._restrict_orientation(desired_orn)
            self.desired_pos = self._restrict_workspace(self.desired_pos)
            return self.desired_pos, self.desired_orn

    def _enforce_5_dof_control(self, abs_target_orn):
        assert len(abs_target_orn) == 3
        abs_target_orn[0] = self.default_orn_x
        abs_target_orn[1] = self.default_orn_y
        return abs_target_orn

    def _enforce_limit_gripper_joint(self, joint_positions, rel_target_orn):
        if rel_target_orn[2] < 0 and joint_positions[-1] - rel_target_orn[2] > self.ul[-1] * 0.8:
            rel_target_orn[2] = 0
        elif rel_target_orn[2] > 0 and joint_positions[-1] - rel_target_orn[2] < self.ll[-1] * 0.8:
            rel_target_orn[2] = 0
        return rel_target_orn

    def _restrict_action_if_contact(self, rel_action_pos, rel_action_orn, state):
        if not np.any(state["contact"]):
            return rel_action_pos, rel_action_orn

        if self.relative_action_control_frame == "world":
            # rel_action_pos = np.linalg.inv(orn_to_matrix(state["tcp_orn"])) @ rel_action_pos
            rel_action_pos, rel_action_orn = to_tcp_frame(rel_action_pos, rel_action_orn, quat_to_euler(state["tcp_orn"]))
        for i in range(3):
            if state["contact"][i]:
                # check opposite signs
                if state["force_torque"][i] * rel_action_pos[i] < 0:
                    rel_action_pos[i] = 0
        for i in range(3):
            if state["contact"][i + 3]:
                # check opposite signs
                if state["force_torque"][i + 3] * rel_action_orn[i] < 0:
                    rel_action_orn[i] = 0
        if self.relative_action_control_frame == "world":
            # rel_action_pos = orn_to_matrix(state["tcp_orn"]) @ rel_action_pos
            rel_action_pos, rel_action_orn = to_world_frame(rel_action_pos, rel_action_orn, quat_to_euler(state["tcp_orn"]))
        return rel_action_pos, rel_action_orn

    def _restrict_orientation(self, desired_orn):
        tcp_orn_mat = orn_to_matrix(desired_orn)
        tcp_x = tcp_orn_mat[:, 0]
        tcp_z = tcp_orn_mat[:, 2]
        if angle_between(tcp_z, np.array([0, 0, -1])) > self.max_orn_z:
            return self.desired_orn
        if np.abs(np.radians(90) - angle_between(tcp_x, np.array([0, 0, 1]))) > self.max_orn_x:
            return self.desired_orn
        return desired_orn

    def _apply_relative_action(self, rel_target_pos, rel_target_orn, tcp_pos, tcp_orn):
        # limit position
        desired_pos = self.desired_pos.copy()
        desired_orn = self.desired_orn.copy()
        for i in range(3):
            if rel_target_pos[i] > 0 and self.desired_pos[i] - tcp_pos[i] < self.relative_pos_clip_threshold:
                desired_pos[i] += rel_target_pos[i]
            elif rel_target_pos[i] < 0 and tcp_pos[i] - self.desired_pos[i] < self.relative_pos_clip_threshold:
                desired_pos[i] += rel_target_pos[i]
        # limit orientation
        rot_diff = quat_to_euler(matrix_to_orn(np.linalg.inv(orn_to_matrix(self.desired_orn)) @ orn_to_matrix(tcp_orn)))
        for i in range(3):
            if rel_target_orn[i] > 0 and rot_diff[i] < self.relative_rot_clip_threshold:
                desired_orn[i] += rel_target_orn[i]
            elif rel_target_orn[i] < 0 and rot_diff[i] > -self.relative_rot_clip_threshold:
                desired_orn[i] += rel_target_orn[i]
        return desired_pos, desired_orn

    def _restrict_workspace(self, target_pos):
        """
        :param target_pos: cartesian target position
        :return: clip target_pos at workspace limits
        """
        return np.clip(target_pos, self.workspace_limits[0], self.workspace_limits[1])
