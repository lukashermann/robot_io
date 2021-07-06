import time

import rospy
import numpy as np
from panda_robot import PandaArm
from franka_dataflow.getch import getch

from robot_io.input_devices.space_mouse import SpaceMouse

print("Initializing node... ")
rospy.init_node("fri_example_joint_position_keyboard")
print("Getting robot state... ")


def clean_shutdown():
    print("\nExiting example.")

mouse = SpaceMouse(act_type='continuous')

rospy.on_shutdown(clean_shutdown)
robot = PandaArm()
robot.move_to_neutral()
has_gripper = robot.get_gripper() is not None
initial_angles = robot.joint_ordered_angles()

force_threshold = [100, 100, 100, 100, 100, 100]  # cartesian force threshold
torque_threshold = [100, 100, 100, 100, 100, 100, 100]  # joint torque threshold
# k_gains = [1200.0, 1000.0, 1000.0, 800.0, 300.0, 200.0, 50.0]
# d_gains = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0]
k_gains = list(np.array([1200.0, 1000.0, 1000.0, 800.0, 300.0, 200.0, 50.0]) / 2)
d_gains = list(np.array([50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0]) / 2)

# increase collision detection thresholds for testing
robot.set_collision_threshold(joint_torques=torque_threshold, cartesian_forces=force_threshold)

cm = robot.get_controller_manager()
if not cm.is_running("franka_ros_interface/effort_joint_impedance_controller"):
    cm.stop_controller(cm.current_controller)
    cm.start_controller("franka_ros_interface/effort_joint_impedance_controller")

# vels = robot.joint_velocities()
# robot.set_joint_positions_velocities(initial_angles, [vels[j] for j in robot.joint_names()])
time.sleep(1)
ctrl_cfg_client = cm.get_current_controller_config_client()
ctrl_cfg_client.set_controller_gains(k_gains, d_gains)

rate = rospy.Rate(100)
print("entering loop")
i = 0
delta = 0.001
j_des = initial_angles
pos, ori = robot.ee_pose()

while not rospy.is_shutdown():
    action = mouse.handle_mouse_events()
    print(i, action)
    mouse.clear_events()

    pos[0] -= action[1] * delta
    pos[1] += action[0] * delta
    pos[2] += action[2] * delta
    status, j = robot.inverse_kinematics(pos, ori)
    if status:
        j_des = j
    robot.set_joint_positions_velocities(j_des, [0] * 7) # impedance control command (see documentation at )
    rate.sleep()
    i += 0.001