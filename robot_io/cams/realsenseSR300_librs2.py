import numpy as np
import cv2

"""
Interface class to get data from RealSense camera.
"""
try:
    import pyrealsense2 as rs
except ImportError:
    print("Try: export LD_LIBRARY_PATH=/<PATH_TO_LIBREALSENSE_2>/librealsense/build/install/lib/:$LD_LIBRARY_PATH")
    raise ImportError


class RealsenseSR300:
    """
    Interface class to get data from RealSense camera.
    """

    def __init__(self, fps=30, img_type="rgb", size=(640, 480)):

        assert img_type in ['rgb', "rgb_depth"]
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
        self.set_rs_options()

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def __del__(self):
        # Stop streaming
        self.pipeline.stop()

    def get_intrinsics(self):
        color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        intr = color_profile.get_intrinsics()
        intr_rgb = dict(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy,
                        cx=intr.ppx, cy=intr.ppy)
        intrinsics = {'rgb': intr_rgb}
        return intrinsics

    def get_projection_matrix(self):
        intr = self.get_intrinsics()['rgb']
        cam_mat = np.array([[intr['fx'], 0, intr['cx'], 0],
                            [0, intr['fy'], intr['cy'], 0],
                            [0, 0, 1, 0]])
        return cam_mat

    def get_camera_matrix(self):
        intr = self.get_intrinsics()['rgb']
        cam_mat = np.array([[intr['fx'], 0, intr['cx']],
                            [0, intr['fy'], intr['cy']],
                            [0, 0, 1]])
        return cam_mat

    def set_rs_options(self, params=None):
        default_params = {
            'white_balance': 3400.0,
            'exposure': 1000.0,
            'brightness': 50.0,
            'contrast': 50.0,
            'saturation': 64.0,
            'sharpness': 50.0,
            'gain': 64.0
        }
        if params is None:
            params = default_params

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

    def get_image(self, crop=False, flip_image=False):
        '''get the the current image as a numpy array'''
        # Wait for a coherent pair of frames: depth and color
        while True:
            try:
                frames = self.pipeline.wait_for_frames()
                break
            except RuntimeError:
                print("Frame didn't arrive within 5000")

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        if crop:
            color_image = color_image[:, 80:560, :]
        if flip_image:
            color_image = color_image[::-1, ::-1, :].copy()
        if self.img_type == 'rgb':
            return color_image, None
        depth_frame = aligned_frames.get_depth_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image.astype(np.float64)
        depth_image *= 0.000125
        if crop:
            depth_image = depth_image[:, 80:560]
        if flip_image:
            depth_image = depth_image[::-1, ::-1].copy()
        return color_image, depth_image

    def project(self, X):
        x = self.get_projection_matrix() @ X
        return x[0:2] / x[2]


def test_cam():
    """
    plotn the camera ouput to test if its working.
    """

    cam = RealsenseSR300(img_type='rgb_depth')
    cam.set_rs_options(params={'white_balance': 3400.0,
                               'exposure': 300.0,
                               'brightness': 50.0,
                               'contrast': 50.0,
                               'saturation': 64.0,
                               'sharpness': 50.0,
                               'gain': 64.0
                               })
    for i in range(100):
        rgb, depth = cam.get_image(flip_image=True, crop=True)
        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.waitKey(1)
        cam.set_rs_options(params={'white_balance': 3400.0,
                                   'exposure': 300.0,
                                   'brightness': 50.0,
                                   'contrast': 50.0,
                                   'saturation': 64.0,
                                   'sharpness': 50.0,
                                   'gain': 64.0
                                   })


if __name__ == "__main__":
    test_cam()
