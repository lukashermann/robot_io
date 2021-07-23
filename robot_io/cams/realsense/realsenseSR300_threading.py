import os
import numpy as np
os.environ["PYRS_INCLUDES"] = "/home/kuka/librealsense_1_12_1/librealsense/include/librealsense/"
import pyrealsense as pyrs
from pyrealsense.constants import rs_option
import threading
import time


class RealsenseSR300:
    def __init__(self, robot=None, fps=30, img_type="rgb"):
        assert img_type in ['rgb', "rgb_depth"]
        self.img_type = img_type
        self.realsense_thread = RealsenseSR300Thread(fps=fps, img_type=img_type, robot=robot)
        self.realsense_thread.start()

    def __del__(self):
        # stop camera and service
        self.realsense_thread.cam.stop()
        self.realsense_thread.serv.stop()

    def set_device_options(self, custom_options):
        self.realsense_thread.set_device_options(custom_options)

    def get_image(self):
        while self.realsense_thread.rgb is None:
            time.sleep(0.01)
        rgb = self.realsense_thread.rgb.copy()
        depth = self.realsense_thread.depth.copy()
        robot_info = self.realsense_thread.robot_info.copy() if self.realsense_thread.robot_info is not None else None

        return rgb, depth, robot_info

    def start_video(self):
        self.realsense_thread.start_video()

    def stop_video(self):
        self.realsense_thread.stop_video()

    def get_video(self):
        rgb_video = self.realsense_thread.rgb_video.copy()
        timestamps = self.realsense_thread.timestamps.copy()
        robot_infos = self.realsense_thread.robot_info_video.copy()
        return rgb_video, robot_infos, timestamps


    def print_info(self):
        for line in self.realsense_thread.cam.get_available_options():
            print(self.realsense_thread.cam.get_device_option_description(line[0][0]))
            print(line)
            print()

    def get_intrinsics(self):

        intr = self.realsense_thread.cam.color_intrinsics
        intr_rgb = dict(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy,
                      cx=intr.ppx, cy=intr.ppy)
        intrinsics = {'rgb': intr_rgb}
        if self.img_type == 'rgb_depth':
            intr =self.realsense_thread.cam.depth_intrinsics
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

    def project(self, X):
        x = self.get_projection_matrix() @ X
        return x[0:2] / x[2]


class RealsenseSR300Thread(threading.Thread):
    def __init__(self, fps=30, img_type="rgb", robot=None):
        # start the service - also available as context manager
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
                              # (rs_option.RS_OPTION_COLOR_WHITE_BALANCE, 4200.0),
                              (rs_option.RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE, 0),
                              (rs_option.RS_OPTION_COLOR_EXPOSURE, 300.0),
                              # (rs_option.RS_OPTION_COLOR_EXPOSURE, 900.0),
                              (rs_option.RS_OPTION_COLOR_BRIGHTNESS, 50.0),
                              (rs_option.RS_OPTION_COLOR_GAMMA, 300.0),
                              (rs_option.RS_OPTION_COLOR_CONTRAST, 50.0),
                              (rs_option.RS_OPTION_F200_LASER_POWER, 15)]



        # self.cam.set_device_options(*zip(*custom_options))
        self.set_rs_defult_options()
        self.robot = robot
        self.robot_info = None
        self.rgb = None
        self.depth = None
        self.record_video = False
        self.rgb_video = []
        self.timestamps = []
        self.robot_info_video = []

        self.fps_controller = FpsController(10)

        threading.Thread.__init__(self)
        self.daemon = True

    def set_device_options(self, custom_options):
        self.cam.set_device_options(*zip(*custom_options))


    def set_rs_defult_options(self):

        params = {
            'white_balance': 3400.0,
            'exposure': 1000.0,
            'brightness': 50.0,
            'contrast': 50.0,
            'saturation': 64.0,
            'sharpness': 50.0,
            'gain': 64.0
        }

        rs_options = [(rs_option.RS_OPTION_COLOR_ENABLE_AUTO_WHITE_BALANCE, 0),
                      (rs_option.RS_OPTION_COLOR_WHITE_BALANCE, params['white_balance']),
                      (rs_option.RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE, 0),
                      (rs_option.RS_OPTION_COLOR_EXPOSURE, params['exposure']),
                      (rs_option.RS_OPTION_COLOR_BRIGHTNESS, params['brightness']),
                      (rs_option.RS_OPTION_COLOR_CONTRAST, params['contrast']),
                      (rs_option.RS_OPTION_COLOR_SATURATION, params['saturation']),
                      (rs_option.RS_OPTION_COLOR_SHARPNESS, params['sharpness']),
                      (rs_option.RS_OPTION_COLOR_GAIN, params['gain']),
                      (rs_option.RS_OPTION_COLOR_GAMMA, 300.0),
                      (rs_option.RS_OPTION_COLOR_HUE, 0.0)]

        self.cam.set_device_options(*zip(*rs_options))

    def start_video(self):
        self.rgb_video.clear()
        self.timestamps.clear()
        self.robot_info_video.clear()
        self.record_video = True

    def stop_video(self):
        self.record_video = False

    def run(self):
        while 1:
            # @timeit
            def rs_xxx():
                self.rgb, self.depth, t, self.robot_info = self.get_image()
                if self.record_video:
                    self.timestamps.append(t)
                    rgb = self.rgb.copy()
                    self.rgb_video.append(rgb)
                    self.robot_info_video.append(self.robot_info.copy())
                self.fps_controller.step()

            rs_xxx()

    def get_image(self):
        self.cam.wait_for_frames()
        t = time.time()

        robot_info = self.robot.get_info() if self.robot is not None else None
        img = np.array(self.cam.color)
        if self.img_type == 'rgb':
            return img, None, t, robot_info
        depth_img = self.cam.dac

        depth_img = depth_img.astype(np.float64)
        depth_img *= 0.000125
        return img, depth_img, t, robot_info


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
        rgb, depth, robot_info = cam.get_image()
        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.imshow("depth", depth)
        cv2.waitKey(1)
        print("robot_info", robot_info)
