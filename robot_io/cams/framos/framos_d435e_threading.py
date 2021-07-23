## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

# Import Numpy for easy array manipulation
import threading
import time

import cv2
import numpy as np
# Import the library
import pyrealsense2 as rs

from robot_io.utils.utils import timeit, FpsController


class FramosD435e:
    """
    Interface class to get data from Framos camera.
    """
    def __init__(self, fps=30, img_type="rgb", resolution=(640, 480), resize_resolution=None, crop_coords=None):
        assert img_type in ['rgb', "rgb_depth"]
        self.camera_thread = _FramosD435e(fps=fps,
                                          img_type=img_type,
                                          resolution=resolution,
                                          resize_resolution=resize_resolution,
                                          crop_coords=crop_coords)
        self.camera_thread.start()
        while self.camera_thread.rgb is None or self.camera_thread.depth is None:
            time.sleep(0.01)

    def get_image(self):
        rgb = self.camera_thread.rgb.copy()
        depth = self.camera_thread.depth.copy()

        return rgb, depth


class _FramosD435e(threading.Thread):
    def __init__(self, fps, img_type, resolution, resize_resolution, crop_coords):
        self.img_type = img_type
        self.rgb = None
        self.depth = None
        self.resolution = resolution
        self.resize_resolution = resize_resolution
        self.crop_coords = crop_coords

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, fps)
        config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.rgb8, fps)

        # Start streaming
        self.profile = self.pipeline.start(config)
        self.color_sensor = self.profile.get_device().first_color_sensor()
        if img_type == 'rgb_depth':
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            # Getting the depth sensor's depth scale (see rs-align example for explanation)
            depth_scale = self.depth_sensor.get_depth_scale()
            print("Depth Scale is: ", depth_scale)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.fps_controller = FpsController(10)

        threading.Thread.__init__(self)
        self.daemon = True

    def __del__(self):
        # Stop streaming
        self.pipeline.stop()

    def run(self):
        while 1:
            # @timeit
            def rs_xxx():
                rgb, depth, t = self.get_image()
                self.rgb = self.crop_and_resize(rgb)
                self.depth = self.crop_and_resize(depth)
                self.fps_controller.step()
            rs_xxx()

    def get_image(self):
        """get the the current image as a numpy array"""
        # Wait for a coherent pair of frames: depth and color
        while True:
            try:
                frames = self.pipeline.wait_for_frames()
                t = time.time()
                break
            except RuntimeError:
                print("Frame didn't arrive within 5000")

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        if self.img_type == 'rgb':
            return color_image, None
        depth_frame = aligned_frames.get_depth_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image.astype(np.float64)
        depth_image *= 0.001

        return color_image, depth_image, t

    def crop_and_resize(self, img):
        if self.crop_coords is not None:
            c = self.crop_coords
            img = img[c[0]: c[1], c[2]:c[3]]
        if self.resize_resolution is not None:
            img = cv2.resize(img, tuple(self.resize_resolution))
        return img


def test_cam():
    # Import OpenCV for easy image rendering
    import cv2
    cam = FramosD435e(img_type='rgb_depth', crop_coords=[0, 480, 80, 560], resize_resolution=(200, 200))
    while 1:
        rgb, depth = cam.get_image()
        cv2.imshow("rgb", rgb[:, :, ::-1])
        depth *= (255 / 1)
        depth = np.clip(depth, 0, 255)
        depth = depth.astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        cv2.imshow("depth", depth)
        cv2.waitKey(1)


if __name__ == "__main__":
    test_cam()
