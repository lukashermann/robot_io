import os
import numpy as np
os.environ["PYRS_INCLUDES"] = "/home/kuka/librealsense_1_12_1/librealsense/include/librealsense/"
import pyrealsense as pyrs
from pyrealsense.constants import rs_option


class RealsenseSR300:
    def __init__(self, fps=30, img_type="rgb"):
        # start the service - also available as context manager
        assert img_type in ['rgb', "rgb_depth"]
        self.serv = pyrs.Service()
        self.serv.start()
        self.img_type = img_type
        # create a device from device id and streams of interest
        self.cam = None
        if self.img_type == 'rgb':
            self.cam = self.serv.Device(device_id=0, streams=[pyrs.stream.ColorStream(fps=fps)])
            custom_options = [(rs_option.RS_OPTION_COLOR_ENABLE_AUTO_WHITE_BALANCE, 0),
                              (rs_option.RS_OPTION_COLOR_WHITE_BALANCE, 3400.0),
                              (rs_option.RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE, 0),
                              # (rs_option.RS_OPTION_COLOR_EXPOSURE, 400.0),
                              # (rs_option.RS_OPTION_COLOR_BRIGHTNESS, 70.0),
                              # (rs_option.RS_OPTION_COLOR_CONTRAST, 70.0)]
                              (rs_option.RS_OPTION_COLOR_EXPOSURE, 300.0),
                              (rs_option.RS_OPTION_COLOR_BRIGHTNESS, 50.0),
                              (rs_option.RS_OPTION_COLOR_GAMMA, 300.0),
                              (rs_option.RS_OPTION_COLOR_CONTRAST, 50.0)]

        else:
            self.cam = self.serv.Device(device_id=0, streams=[
                pyrs.stream.ColorStream(fps=fps, width=640, height=480),
                pyrs.stream.DepthStream(fps=fps),
                pyrs.stream.DACStream(fps=fps),
                pyrs.stream.InfraredStream(fps=fps)])

            custom_options = [(rs_option.RS_OPTION_COLOR_ENABLE_AUTO_WHITE_BALANCE, 0),
                              (rs_option.RS_OPTION_COLOR_WHITE_BALANCE, 3400.0),
                              (rs_option.RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE, 0),
                              (rs_option.RS_OPTION_COLOR_EXPOSURE, 300.0),
                              (rs_option.RS_OPTION_COLOR_BRIGHTNESS, 50.0),
                              (rs_option.RS_OPTION_COLOR_GAMMA, 300.0),
                              (rs_option.RS_OPTION_COLOR_CONTRAST, 50.0),
                              (rs_option.RS_OPTION_F200_LASER_POWER, 15)]
        self.cam.set_device_options(*zip(*custom_options))

    def __del__(self):
        # stop camera and service
        self.cam.stop()
        self.serv.stop()

    def print_info(self):
        for line in self.cam.get_available_options():
            print(self.cam.get_device_option_description(line[0][0]))
            print(line)
            print()

    def get_intrinsics(self):

        intr = self.cam.color_intrinsics
        intr_rgb = dict(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy,
                      cx=intr.ppx, cy=intr.ppy)
        intrinsics = {'rgb': intr_rgb}
        if self.img_type == 'rgb_depth':
            intr =self.cam.depth_intrinsics
            intr_depth = dict(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy,
                              cx=intr.ppx, cy=intr.ppy)
            intrinsics['depth'] = intr_depth
        return intrinsics

    def get_projection_matrix(self):
        intr = self.get_intrinsics()['rgb']
        cam_mat  = np.array([[intr['fx'], 0, intr['cx'], 0],
                            [0, intr['fy'], intr['cy'], 0],
                            [0, 0, 1, 0]])
        return cam_mat

    def get_camera_matrix(self):
        intr = self.get_intrinsics()['rgb']
        cam_mat  = np.array([[intr['fx'], 0, intr['cx']],
                            [0, intr['fy'], intr['cy']],
                            [0, 0, 1]])
        return cam_mat

    def get_image(self, crop=False, flip_image=False):
        self.cam.wait_for_frames()

        img = np.array(self.cam.color)
        if crop:
            img = img[:, 80:560, :]
        if flip_image:
            img = img[::-1, ::-1, :]
        if self.img_type == 'rgb':
            return img, None
        depth_img = self.cam.dac
        if flip_image:
            depth_img = depth_img[::-1, ::-1]

        depth_img = depth_img.astype(np.float64)
        depth_img *= 0.000125
        return img, depth_img

    def project(self, X):
        x = self.get_projection_matrix() @ X
        return x[0:2] / x[2]


if __name__ == "__main__":
    import cv2

    cam = RealsenseSR300(img_type='rgb_depth')
    print(cam.get_projection_matrix())
    T_tcp_cam = np.array([[0.99987185, -0.00306941, -0.01571176, 0.00169436],
                          [-0.00515523, 0.86743151, -0.49752989, 0.11860651],
                          [0.015156, 0.49754713, 0.86730453, -0.18967231],
                          [0., 0., 0., 1.]])
    K = np.array([[617.89, 0, 315.2, 0],
                  [0, 617.89, 245.7, 0],
                  [0, 0, 1, 0]])


    def project(K, X):
        x = K @ X
        return x[0:2] / x[2]


    while 1:
        rgb, depth = cam.get_image(crop=False)
        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.imshow("depth", depth)
        cv2.waitKey(1)
