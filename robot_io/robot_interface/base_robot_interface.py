from enum import Enum

import cv2
from scipy.spatial.transform.rotation import Rotation as R
import numpy as np

from robot_io.utils.utils import euler_to_quat


# Enum's are not interpreted as numerical values automatically, as such
# we don't want to have them in actions, for this use Enum.value.
class GripperState(Enum):
    OPEN = 1
    CLOSED = -1


class GripperInterface:
    @staticmethod
    def to_gripper_state(gs):
        if gs == "open":
            return GripperState.OPEN
        elif gs == "closed":
            return GripperState.CLOSED
        elif gs in (GripperState.OPEN, GripperState.CLOSED):
            return gs
        else:
            raise ValueError(f"Invalid gripper state {gs} must be GripperState enum")

    @staticmethod
    def toggle(gs):
        if gs == GripperState.OPEN:
            return GripperState.CLOSED
        elif gs == GripperState.CLOSED:
            return GripperState.OPEN
        else:
            raise ValueError(f"gripper state must be GripperState Enum was {gs}.")


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
        Get the proprioceptive robot state consisting of the following entries:
        - tcp_pos: Tcp position as position (x,y,z)
        - tcp_orn: Tcp orientation as quaternion (x,y,z,w)
        - joint_positions: Joint angles in rad.
        - gripper_opening_width: Gripper opening width in meter.
        - force_torque: Estimated external wrench (force, torque) acting on tcp frame in [N,N,N,Nm,Nm,Nm].
        - contact: Indicates which contact level is activated in which Cartesian dimension.
            After contact disappears, the value turns to zero.

        Returns:
            Dictionary with full robot state.
        """
        raise NotImplementedError

    def get_tcp_pose(self):
        """
        Get tcp pose in base frame as homogeneous matrix (4x4 np.ndarray).

        Returns:
            tcp_pose_matrix (O_T_EE).
        """
        raise NotImplementedError

    def get_tcp_pos_orn(self):
        """
        Get tcp pose in base frame as position and orientation.

        Returns:
             tcp_pos: position (x,y,z).
             tcp_orn: orientation quaternion (x,y,z,w).
        """
        raise NotImplementedError

    def move_cart_pos(self, target_pos, target_orn, ref="abs", path="ptp", blocking=True, impedance=False):
        """
        Move robot to cartesian pose.

        Args:
            target_pos: Target position (x,y,z).
            target_orn: Target orientation as quaternion (x,y,z,w) or euler_angles (α,β,γ).
            ref: Reference frame. "abs" for absolute cartesian poses, "rel" for relative cartesian poses.
            path: Path planning method. "ptp" for a movement linear in joint space,
                  "lin" for a movement linear in cartesian space. Note that it might not be possible to reach the whole
                  workspace space with a linear motion from an arbitrary configuration.
            blocking: If True, wait until target pose is reached.
            impedance: If True, use impedance control (if available).
        """
        raise NotImplementedError

    def move_joint_pos(self, joint_positions, blocking=True):
        """
        Move robot to absolute joint positions, blocking.

        Args:
            joint_positions: (j1, ..., jn) in rad.
            blocking: If True, wait until target position is reached.
        """
        raise NotImplementedError

    def move_joint_vel(self, joint_velocities, blocking=True):
        """
        Move robot asynchronously with joint velocities.

        Args:
            joint_velocities: (j1, ..., jn).
            blocking: If True, wait until target position is reached.
        """
        raise NotImplementedError

    def abort_motion(self):
        """
        Stop the execution of the current motion.
        """
        raise NotImplementedError

    def close_gripper(self, blocking=False):
        """
        Close fingers of the gripper. If the fingers are already closed, this action has no effect.

        Args:
            blocking: If True, wait for gripper action to be finished.
        """
        raise NotImplementedError

    def open_gripper(self, blocking=False):
        """
        Open fingers of the gripper. If the fingers are already open, this action has no effect.

        Args:
            blocking: If True, wait for gripper action to be finished.
        """
        raise NotImplementedError

    def reached_position(self, target_pos, target_orn, cart_threshold=0.005, orn_threshold=0.05):
        """
        Check if robot has reached a target pose (e.g. after sending an asynchronous movement command).
        Try increasing the thresholds if this method never returns True.

        Args:
            target_pos: (x,y,z)
            target_orn: Quaternion (x,y,z,w) | Euler_angles (α,β,γ).
            cart_threshold: Cartesian position error threshold for euclidean distance
                between current_pos and target_pos, in meter
            orn_threshold: Angular error threshold, in radian.

        Returns:
            True if reached pose, False otherwise.
        """
        if len(target_orn) == 3:
            target_orn = euler_to_quat(target_orn)
        curr_pos, curr_orn = self.get_tcp_pos_orn()
        pos_error = np.linalg.norm(target_pos - curr_pos)
        orn_error = np.linalg.norm((R.from_quat(target_orn) * R.from_quat(curr_orn).inv()).as_rotvec())
        return pos_error < cart_threshold and orn_error < orn_threshold

    def reached_joint_state(self, target_state, threshold=0.001):
        """
        Check if robot has reached a target joint state (e.g. after sending an asynchronous movement command).
        Try increasing the thresholds if this method never returns True.

        Args:
            target_state: (j1, ..., jn)
            threshold: In rad.

        Returns:
            True if reached state, False otherwise.
        """
        curr_pos = self.get_state()['joint_positions']
        offset = np.sum(np.abs((np.array(target_state) - curr_pos)))
        return offset < threshold

    def visualize_joint_states(self):
        """
        Visualize the robot's joint states (left border is lower bound, right border is upper bound).
        """
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

    def _visualize_external_forces(self, contact, collision, canvas_width=500):
        """
        Display the external forces (x,y,z) and torques (a,b,c) of the tcp frame.

        Args:
            contact: TODO:
            collision: TODO:
            canvas_width: Display width in pixel.
        """
        canvas = np.ones((300, canvas_width, 3))
        forces = self.get_state()["force_torque"]

        left = 10
        right = canvas_width - left
        width = right - left
        height = 30
        y = 10
        for i, (lcol, lcon, f, ucon, ucol) in enumerate(zip(-collision, -contact, forces, contact, collision)):
            cv2.rectangle(canvas, [left, y], [right, y + height], [0, 0, 0], thickness=2)
            force_bar_pos = int(left + width * (f - lcol) / (ucol - lcol))
            cv2.line(canvas, [force_bar_pos, y], [force_bar_pos, y + height], thickness=4, color=[0, 0, 1])
            ucon_bar_pos = int(left + width * (ucon - lcol) / (ucol - lcol))
            cv2.line(canvas, [ucon_bar_pos, y], [ucon_bar_pos, y + height], thickness=2, color=[1, 0, 0])
            lcon_bar_pos = int(left + width * (lcon - lcol) / (ucol - lcol))
            cv2.line(canvas, [lcon_bar_pos, y], [lcon_bar_pos, y + height], thickness=2, color=[1, 0, 0])

            y += height + 10
        cv2.imshow("external_forces", canvas)
        cv2.waitKey(1)
