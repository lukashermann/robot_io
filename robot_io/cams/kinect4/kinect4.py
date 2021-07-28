import json

import open3d as o3d
import numpy as np
import cv2
import time
import threading
from pathlib import Path

from robot_io.cams.camera import Camera
from robot_io.utils.utils import FpsController, timeit


class Kinect4(Camera):
    def __init__(self,
                 device=0,
                 align_depth_to_color=True,
                 config_path='config/config_kinect4_1080p.json',
                 params_file_path="config/kinect4_params_1080p.npz",
                 undistort_image=True,
                 resize_resolution=None,
                 crop_coords=None,
                 fps=30):
        super().__init__()
        data = np.load((Path(__file__).parent / params_file_path).as_posix(), allow_pickle=True)
        if config_path is not None:
            full_path = (Path(__file__).parent / config_path).as_posix()
            config = o3d.io.read_azure_kinect_sensor_config(full_path)
        else:
            config = o3d.io.AzureKinectSensorConfig()

        self.dist_coeffs = data['dist_coeffs']
        self.camera_matrix = data['camera_matrix']
        self.projection_matrix = data['projection_matrix']
        self.intrinsics = data['intrinsics'].item()
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, R=np.eye(3), \
                newCameraMatrix=self.camera_matrix, size=(self.intrinsics['width'], self.intrinsics['height']), m1type= cv2.CV_16SC2)
        self.device = device
        self.align_depth_to_color = align_depth_to_color
        self.config_path = config_path
        self.undistort_image = undistort_image
        self.resize_resolution = resize_resolution
        self.crop_coords = crop_coords
        self.fps = fps

        self.align_depth_to_color = align_depth_to_color
        self.sensor = o3d.io.AzureKinectSensor(config)
        if not self.sensor.connect(device):
            raise RuntimeError('Failed to connect to sensor')

    def get_intrinsics(self):
        return self.intrinsics

    def get_projection_matrix(self):
        return self.projection_matrix

    def get_camera_matrix(self):
        return self.camera_matrix

    def _get_image(self):
        rgbd = None
        while rgbd is None:
            rgbd = self.sensor.capture_frame(self.align_depth_to_color)
        rgb = np.asarray(rgbd.color)
        depth = (np.asarray(rgbd.depth)).astype('float') / 1000
        if self.undistort_image:
            rgb = cv2.remap(rgb, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
            depth = cv2.remap(depth, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        return rgb, depth


def select_roi():
    cam = Kinect4(0)
    resolution = (200, 150)
    while True:
        for i in range(10):
            rgb, depth = cam.get_image()
            cv2.imshow("rgb", rgb[:, :, ::-1])
            cv2.imshow("depth", depth)
            cv2.waitKey(1)
        r = cv2.selectROI("rgb", rgb[:, :, ::-1])
        x, y = int(r[0]), int(r[1])
        width, height = int(r[2]), int(r[3])
        center = (x + width // 2, y + height // 2)
        ratio = resolution[0] / resolution[1]
        if width / height > ratio:
            height = width / ratio
        else:
            width = height * ratio
        height = int(np.round(height))
        width = int(np.round(width))

        mask = np.zeros_like(depth)
        mask[center[1] - height // 2: center[1] + height // 2, center[0] - width // 2: center[0] + width // 2] = 1
        rgb[np.where(mask == 0)] = 0
        depth[np.where(mask == 0)] = 0

        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.imshow("depth", depth)
        print("Press ENTER to finish selection, press c button to redo.")
        k = cv2.waitKey(0) % 256
        if k == 13:
            print("Image coordinates: ", center[1] - height // 2, center[1] + height // 2, center[0] - width // 2, center[0] + width // 2)
            return
        else:
            continue


def test_camera():
    # cam = Kinect4(0, crop_coords=(301, 623, 516, 946), resize_resolution=(200, 150))
    cam = Kinect4(0)
    while True:
        rgb, depth = cam.get_image()
        cv2.imshow("depth", depth)
        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.waitKey(1)


if __name__ == "__main__":
    # select_roi()
    test_camera()
