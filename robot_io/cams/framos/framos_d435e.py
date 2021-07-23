## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

# Import Numpy for easy array manipulation
import numpy as np
# Import the library
import pyrealsense2 as rs

from robot_io.cams.kinect4.kinect4_threading import timeit


class FramosD435e:
    """
    Interface class to get data from Framos camera.
    """

    def __init__(self, fps=30, img_type="rgb"):

        assert img_type in ['rgb', "rgb_depth"]
        self.img_type = img_type

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, fps)

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

    def __del__(self):
        # Stop streaming
        self.pipeline.stop()

    @timeit
    def get_image(self):
        """get the the current image as a numpy array"""
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
        if self.img_type == 'rgb':
            return color_image, None
        depth_frame = aligned_frames.get_depth_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image.astype(np.float64)
        depth_image *= 0.001
        return color_image, depth_image


def test_cam():
    # Import OpenCV for easy image rendering
    import cv2
    cam = FramosD435e(img_type='rgb_depth')
    while 1:
        rgb, depth = cam.get_image()
        cv2.imshow("rgb", rgb[:, :, ::-1])
        # depth *= (255 / 4)
        # depth = np.clip(depth, 0, 255)
        # depth = depth.astype(np.uint8)
        # depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        # cv2.imshow("depth", depth)
        cv2.waitKey(1)


if __name__ == "__main__":
    test_cam()