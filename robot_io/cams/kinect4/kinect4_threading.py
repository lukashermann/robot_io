import open3d as o3d
import numpy as np
import cv2
import time
import threading
from pathlib import Path

from robot_io.utils.utils import FpsController, timeit


class Kinect4:
    def __init__(self,
                 device=0,
                 align_depth_to_color=True,
                 config_path='config/default_config_kinect4.json',
                 params_file_path="config/kinect4_params_720p.npz",
                 undistort_image=True,
                 resize_resolution=None,
                 crop_coords=None,
                 fps=30):
        data = np.load(Path(__file__).parent / params_file_path, allow_pickle=True)
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
        if config_path is not None:
            self.config = o3d.io.read_azure_kinect_sensor_config((Path(__file__).parent / config_path).as_posix())
        else:
            self.config = o3d.io.AzureKinectSensorConfig()

        self.align_depth_to_color = align_depth_to_color

        self.kinect_thread = None
        self.start_camera_thread()

    def __del__(self):
        self.stop_camera_thread()

    @timeit
    def start_camera_thread(self):
        self.kinect_thread = Kinect4Thread(self.fps,
                                           self.map1,
                                           self.map2,
                                           self.config,
                                           self.device,
                                           self.align_depth_to_color,
                                           self.undistort_image,
                                           self.resize_resolution,
                                           self.crop_coords)
        self.kinect_thread.start()
        # while not self.kinect_thread.is_alive():
        #     time.sleep(0.001)

    @timeit
    def stop_camera_thread(self):
        if self.kinect_thread is not None:
            self.kinect_thread.flag_exit = True
            while self.kinect_thread.is_alive():
                time.sleep(0.001)
            del self.kinect_thread
            self.kinect_thread = None

    def get_intrinsics(self):
        return self.intrinsics

    def get_projection_matrix(self):
        return self.projection_matrix

    def get_image(self):
        while self.kinect_thread.rgb is None or self.kinect_thread.depth is None:
            time.sleep(0.01)
        rgb = self.kinect_thread.rgb.copy()
        depth = self.kinect_thread.depth.copy()

        return rgb, depth


class Kinect4Thread(threading.Thread):
    def __init__(self,
                 fps,
                 map1,
                 map2,
                 config,
                 device,
                 align_depth_to_color,
                 undistort_image,
                 resize_resolution,
                 crop_coords):
        self.flag_exit = False
        self.align_depth_to_color=align_depth_to_color
        self.sensor = o3d.io.AzureKinectSensor(config)
        if not self.sensor.connect(device):
            raise RuntimeError('Failed to connect to sensor')
        self.map1, self.map2 = map1, map2
        self.rgb = None
        self.depth = None
        self.undistort_image = undistort_image
        self.resize_resolution = resize_resolution
        self.crop_coords = crop_coords
        self.fps_controller = FpsController(fps)

        self.timestamps = []

        threading.Thread.__init__(self)
        self.daemon = True

    def run(self):
        while not self.flag_exit:
            # @timeit
            def k4_xxx():
                rgb, depth, t = self.get_image()
                if self.undistort_image:
                    rgb = cv2.remap(rgb, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
                    depth = cv2.remap(depth, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
                self.rgb = self.crop_and_resize(rgb)
                self.depth = self.crop_and_resize(depth)

                self.fps_controller.step()
            k4_xxx()

    def crop_and_resize(self, img):
        if self.crop_coords is not None:
            c = self.crop_coords
            img = img[c[0]: c[1], c[2]:c[3]]
        if self.resize_resolution is not None:
            img = cv2.resize(img, tuple(self.resize_resolution))
        return img

    def get_image(self):
        rgbd = None
        while rgbd is None:
            rgbd = self.sensor.capture_frame(self.align_depth_to_color)
        t = time.time()
        rgb = np.asarray(rgbd.color)
        depth = (np.asarray(rgbd.depth)).astype('float') / 1000
        return rgb, depth, t


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
    cam = Kinect4(0, crop_coords=(301, 623, 516, 946), resize_resolution=(200, 150))
    while True:
        rgb, depth = cam.get_image()
        cv2.imshow("depth", depth)
        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.waitKey(1)


if __name__ == "__main__":
    # select_roi()
    test_camera()
