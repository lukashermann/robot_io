import rospy
from panda_robot import PandaArm
from panda_env import PandaEnv
from robot_io.input_devices.vr_input_panda import VrInput
from robot_io.panda_control.vr_recorder import VrRecorder


def clean_shutdown():
    print("\nExiting example.")


print("Initializing node... ")
rospy.init_node("panda_vr_control_joint_impedance")
print("Getting robot state... ")

robot = PandaArm()
vr_input = VrInput(robot)
panda_env = PandaEnv(robot)

rospy.on_shutdown(clean_shutdown)
rate = rospy.Rate(100)

obs = panda_env.reset()
recorder = VrRecorder("/tmp/recording")

vr_input.calibrate_vr_coord_system()
vr_input.wait_for_start_button()

while not rospy.is_shutdown():
    action, record_info = vr_input.get_vr_action()
    obs, _, _, _ = panda_env.step(action)
    recorder.step(obs, record_info)
    panda_env.render()
    rate.sleep()
