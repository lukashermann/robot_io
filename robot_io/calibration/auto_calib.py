import numpy as np
import cv2
import cv2.aruco as aruco
import math
import time
from robot_io.robot_interface.iiwa_interface import IIWAInterface
from robot_io.cams.kinect4.kinect4 import Kinect4
from robot_io.calibration.marker_detection import detect_marker

from math import pi


def work_position(controller):
    pos = [0, -0.56, 0.26]
    orn = [pi, 0, pi / 2]
    controller.move_async_cart_pos_abs_ptp(pos, orn)


if __name__ == "__main__":
    cam = Kinect4()
    iiwa = IIWAInterface(use_impedance=True)
    # work_position(iiwa)
    max_angles = [165, 115, 165, 115, 165, 115, 170]
    min_angles = [-x for x in max_angles]
    dcm_list = []
    reset = True
    detect_counter = 0
    NUM_JOINTS = 7
    res_angles = np.zeros(NUM_JOINTS)
    last_joint = NUM_JOINTS - 1
    shift_degree = 10

    joint_angles = np.rad2deg(iiwa.get_state()['joint_positions'])
    if joint_angles[last_joint] < 0:
        res_angles[last_joint] = shift_degree
    else:
        res_angles[last_joint] = -shift_degree

    while True:
        rgb, depth = cam.get_image()
        detected, dcm = detect_marker(rgb, 0.1, marker_id=0, camera_matrix=cam.get_camera_matrix(),
                                    marker_dict=aruco.DICT_4X4_250)
        joint_angles = np.rad2deg(iiwa.get_state()['joint_positions'])

        # Detect the marker two times with small end-effector pose change -> Marker is discovered!
        if detected:

            detect_counter += 1
            if detect_counter > 2:
                print("Marker is discovered")
                print("dcm", dcm)
                exit()

            next_joint_angles = joint_angles + res_angles

        else:
            # Rotate end-effector to the closest extreme
            if reset or detect_counter != 0:
                reset = False
                detect_counter = 0
                if joint_angles[last_joint] < 0:
                    joint_angles[last_joint] = min_angles[last_joint]
                    res_angles[last_joint] = shift_degree
                else:
                    joint_angles[last_joint] = max_angles[last_joint]
                    res_angles[last_joint] = -shift_degree
                next_joint_angles = joint_angles

            # Rotate end-effector gradually to the other extreme
            else:
                next_joint_angles = joint_angles + res_angles

        next_joint_angles = np.clip(next_joint_angles, min_angles, max_angles).tolist()
        iiwa.move_async_joint_pos(np.deg2rad(next_joint_angles))
        time.sleep(3)







