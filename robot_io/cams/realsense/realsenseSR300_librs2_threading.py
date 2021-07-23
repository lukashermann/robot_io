import numpy as np
import cv2

from robot_io.utils.utils import FpsController

"""
Interface class to get data from RealSense camera.
"""
try:
    import pyrealsense2 as rs
except ImportError:
    print("Try: export LD_LIBRARY_PATH=/home/kuka/librealsense_2/librealsense/build/install/lib/:$LD_LIBRARY_PATH")
    raise ImportError
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

    # def set_device_options(self, custom_options):
    #     self.realsense_thread.set_device_options(custom_options)

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

    def get_intrinsics(self):
        return self.realsense_thread.get_intrinsics()

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
    def __init__(self, fps=30, img_type="rgb", robot=None, size=(640, 480)):

        self.img_type = img_type

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, size[0], size[1], rs.format.rgb8, fps)
        if self.img_type == "rgb_depth":
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        self.size = size

        # Start streaming
        self.profile = self.pipeline.start(config)
        self.color_sensor = self.profile.get_device().first_color_sensor()
        if img_type == 'rgb_depth':
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.set_rs_default_options()

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)


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
    #
    # def set_device_options(self, custom_options):
    #     self.cam.set_device_options(*zip(*custom_options))

    def get_intrinsics(self):
        color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        intr = color_profile.get_intrinsics()
        intr_rgb = dict(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy,
                        cx=intr.ppx, cy=intr.ppy)
        intrinsics = {'rgb': intr_rgb}
        return intrinsics

    def set_rs_default_options(self):
        params = {
            'white_balance': 3400.0,
            'exposure': 1000.0,
            'brightness': 50.0,
            'contrast': 50.0,
            'saturation': 64.0,
            'sharpness': 50.0,
            'gain': 64.0
        }
        self.color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        self.color_sensor.set_option(rs.option.white_balance, params['white_balance'])
        self.color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        self.color_sensor.set_option(rs.option.exposure, params['exposure'])
        self.color_sensor.set_option(rs.option.brightness, params['brightness'])
        self.color_sensor.set_option(rs.option.contrast, params['contrast'])
        self.color_sensor.set_option(rs.option.saturation, params['saturation'])
        self.color_sensor.set_option(rs.option.sharpness, params['sharpness'])
        self.color_sensor.set_option(rs.option.gain, params['gain'])
        self.color_sensor.set_option(rs.option.gamma, 300)
        self.color_sensor.set_option(rs.option.hue, 0.0)
        # self.depth_sensor.set_option(rs.option.laser_power, 0.0)

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
        frames = self.pipeline.wait_for_frames()
        t = time.time()

        robot_info = self.robot.get_info() if self.robot is not None else None

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        if self.img_type == 'rgb':
            return color_image, None, t, robot_info
        depth_frame = aligned_frames.get_depth_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image.astype(np.float64)
        depth_image *= 0.000125

        return color_image, depth_image, t, robot_info


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
