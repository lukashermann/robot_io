import time
import logging
import threading

import hydra
import numpy as np

from multiprocessing import RLock
from robot_io.utils.utils import FpsController, timeit


log = logging.getLogger(__name__)


class ThreadedCamera:
    def __init__(self, camera_cfg):
        self._camera_thread = _CameraThread(camera_cfg)
        self._camera_thread.start()
        self._last_frame = 0
    
    @property
    def resolution(self):
        return self._camera_thread.camera.resolution
    @property    
    def crop_coords(self):
        return self._camera_thread.camera.crop_coords
    @property    
    def resize_resolution(self):
        return self._camera_thread.camera.resize_resolution
    @property    
    def name(self):
        return self._camera_thread.camera.name


    @property
    def frame_count(self):
        return self._camera_thread.frame_count

    def get_image(self, fetch_new=True):
        while fetch_new and self._camera_thread.frame_count == self._last_frame:
            time.sleep(0.01)
     
        self.frame_count, rgb, depth = self._camera_thread.get_image()
        return rgb, depth

    def get_intrinsics(self):
        return self._camera_thread.camera.get_intrinsics()

    def get_projection_matrix(self):
        return self._camera_thread.camera.get_projection_matrix()

    def get_camera_matrix(self):
        return self._camera_thread.camera.get_camera_matrix()

    def get_dist_coeffs(self):
        return self._camera_thread.camera.get_dist_coeffs()

    def compute_pointcloud(self, depth_img, rgb_img=None, far_val=10, homogeneous=False):
        return self._camera_thread.camera.compute_pointcloud(depth_img, rgb_img, far_val, homogeneous)

    def view_pointcloud(self, pointcloud):
        return self._camera_thread.camera.view_pointcloud(pointcloud)

    def revert_crop_and_resize(self, img):
        return self._camera_thread.camera.revert_crop_and_resize(img)

    def get_extrinsic_calibration(self, robot_name):
        return self._camera_thread.camera.get_extrinsic_calibration(robot_name)

    def deproject(self, point, depth, homogeneous=False):
        return self._camera_thread.camera.deproject(point, depth, homogeneous)

    def __del__(self):
        log.info("Closing camera.")
        self._camera_thread.flag_exit = True
        self._camera_thread.join()


class _CameraThread(threading.Thread):
    def __init__(self, camera_cfg):
        threading.Thread.__init__(self)
        self.camera = hydra.utils.instantiate(camera_cfg)
        self.daemon = True
        self.fps_controller = FpsController(camera_cfg.fps)
        self.flag_exit = False
        self.rgb = None
        self.depth = None
        self._lock = RLock()
        self._frame_count = -1

    def run(self):
        while not self.flag_exit:
            rgb, depth = self.camera.get_image()
            with self._lock:
                self.rgb  = rgb
                self.depth = depth
                self._frame_count += 1

            self.fps_controller.step()

    @property
    def frame_count(self):
        with self._lock:
            return self._frame_count

    def get_image(self):
        with self._lock:
            return self._frame_count, self.rgb, self.depth


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import cv2
    cfg = OmegaConf.load("../conf/cams/gripper_cam/framos_highres.yaml")
    cam = ThreadedCamera(cfg)

    while True:
        rgb, depth = cam.get_image()
        # pc = cam.compute_pointcloud(depth, rgb)
        # cam.view_pointcloud(pc)
        cv2.imshow("depth", depth)
        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.waitKey(1)
