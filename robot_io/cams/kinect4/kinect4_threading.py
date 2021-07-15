import open3d as o3d
import numpy as np
import cv2
import time
import threading
from pathlib import Path

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


class FpsController:
    def __init__(self, freq):
        self.loop_time = 1.0 / freq
        self.prev_time = time.time()

    def step(self):
        current_time = time.time()
        delta_t = current_time - self.prev_time
        if delta_t < self.loop_time:
            time.sleep(self.loop_time - delta_t)
        self.prev_time = time.time()


class Kinect4:
    def __init__(self, device=0, align_depth_to_color=True, config_path='config/default_config_kinect4.json', undistort_video=True):
        data = np.load(Path(__file__).parent / "config/kinect4_params.npz", allow_pickle=True)
        self.dist_coeffs = data['dist_coeffs']
        self.camera_matrix = data['camera_matrix']
        self.projection_matrix = data['projection_matrix']
        self.intrinsics = data['intrinsics'].item()
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, R=np.eye(3), \
                newCameraMatrix=self.camera_matrix, size=(self.intrinsics['width'], self.intrinsics['height']), m1type= cv2.CV_16SC2)
        self.device = device
        self.align_depth_to_color = align_depth_to_color
        self.config_path = config_path
        self.undistort_video = undistort_video
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
        self.kinect_thread = Kinect4Thread(self.map1, self.map2, self.config, self.device, self.align_depth_to_color,
                                           self.config_path, self.undistort_video)
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

    def get_image(self, undistorted=True):
        while self.kinect_thread.rgb is None:
            time.sleep(0.01)
        rgb = self.kinect_thread.rgb.copy()
        depth = self.kinect_thread.depth.copy()
        if undistorted:
            rgb = cv2.remap(rgb, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
            depth = cv2.remap(depth, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        return rgb, depth

    def start_video(self):
        self.kinect_thread.start_video()

    def stop_video(self):
        self.kinect_thread.stop_video()

    def get_video(self):
        rgb_video = self.kinect_thread.rgb_video.copy()
        depth_video = self.kinect_thread.depth_video.copy()
        timestamps = self.kinect_thread.timestamps.copy()

        return rgb_video, depth_video, timestamps


class Kinect4Thread(threading.Thread):
    def __init__(self, map1, map2, config, device=0, align_depth_to_color=True, config_path='config/default_config_kinect4.json', undistort_video=True):
        self.flag_exit = False
        self.align_depth_to_color=align_depth_to_color
        self.sensor = o3d.io.AzureKinectSensor(config)
        if not self.sensor.connect(device):
            raise RuntimeError('Failed to connect to sensor')
        self.map1, self.map2 = map1, map2
        self.rgb = None
        self.depth = None
        self.undistort_video = undistort_video
        self.record_video = False
        self.rgb_video = []
        self.depth_video = []

        self.fps_controller = FpsController(10)

        self.timestamps = []

        threading.Thread.__init__(self)
        self.daemon = True

    def start_video(self):
        self.rgb_video.clear()
        self.depth_video.clear()
        self.timestamps.clear()
        self.record_video = True

    def stop_video(self):
        self.record_video = False

    def run(self):
        while not self.flag_exit:
            # @timeit
            def k4_xxx():
                self.rgb, self.depth, t = self.get_image()
                if self.record_video:
                    if self.undistort_video:
                        rgb = cv2.remap(self.rgb.copy(), self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
                        depth = cv2.remap(self.depth.copy(), self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
                    else:
                        rgb = self.rgb.copy()
                        depth = self.depth.copy()
                    self.timestamps.append(t)
                    self.rgb_video.append(rgb)
                    self.depth_video.append(depth)
                self.fps_controller.step()
            k4_xxx()

    def get_image(self):
        rgbd = None
        while rgbd is None:
            rgbd = self.sensor.capture_frame(self.align_depth_to_color)
        t = time.time()
        rgb = np.asarray(rgbd.color)
        depth = (np.asarray(rgbd.depth)).astype('float') / 1000
        return rgb, depth, t


if __name__ == "__main__":
    num_cams = 2
    cams = [Kinect4(device=i) for i in range(num_cams)]
    for i in range(1000):
        for i, cam in enumerate(cams):
            rgb, depth = cam.get_image(undistorted=False)
            # cv2.imshow("win", depth)
            cv2.imshow(f"win_{i}", rgb[:, :, ::-1])
        cv2.waitKey(1)
