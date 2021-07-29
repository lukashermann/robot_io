import numpy as np
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R


def detect_marker(rgb, marker_size, camera_matrix, marker_id=None, dist_coeffs=np.zeros(12),
                  marker_dict=aruco.DICT_4X4_250):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # set dictionary size depending on the aruco marker selected
    aruco_dict = aruco.Dictionary_get(marker_dict)
    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    # lists of ids and the corners belonging to each marker_id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # check if the ids list is not empty
    # if no check is added the code will crash
    if ids is None:
        print("no marker detected")
        # code to show 'No Ids' when no markers are found
        cv2.putText(rgb, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("win2", rgb[:, :, ::-1])
        cv2.waitKey(1)

        return False, None

    if marker_id is not None:
        if marker_id not in ids:
            return False, None
        else:
            pos = np.where(ids == marker_id)[0][0]
            corners = corners[pos: pos + 1]
            # estimate pose of each marker and return the values
            # rvet and tvec-different from camera coefficients
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
            # (rvec-tvec).any() # get rid of that nasty numpy value array error

            # draw axis for the aruco markers
            aruco.drawAxis(rgb, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.1)

            # draw a square around the markers
            aruco.drawDetectedMarkers(rgb, corners)

            # code to show ids of the marker found
            strg = ''
            for i in range(0, ids.size):
                strg += str(ids[i][0]) + ', '

            cv2.putText(rgb, "Id: " + strg, (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print("marker detected with marker_id {}".format(strg))
            #
            cv2.imshow("win2", rgb[:,:,::-1])
            cv2.waitKey(1)

            r = R.from_rotvec(rvec[0])
            dcm = np.eye(4)
            dcm[:3, 3] = tvec
            dcm[:3, :3] = r.as_matrix()
            return True, dcm

    return False, None


if __name__ == "__main__":
    from robot_io.cams.kinect4.kinect4 import Kinect4
    cam = Kinect4()
    while True:
        rgb, depth = cam.get_image()
        detect_marker(rgb, 11, 0.1, camera_matrix=cam.get_camera_matrix(), marker_dict=aruco.DICT_4X4_250)