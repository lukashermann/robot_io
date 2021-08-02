import threading
import time

import hydra
import numpy as np

from robot_io.utils.utils import FpsController, timeit


class ThreadedCamera:
    def __init__(self, camera_cfg):
        self._camera_thread = _CameraThread(camera_cfg)
        self._camera_thread.start()

    def get_image(self):
        while self._camera_thread.rgb is None or self._camera_thread.depth is None:
            time.sleep(0.01)
        rgb = self._camera_thread.rgb.copy()
        depth = self._camera_thread.depth.copy()

        return rgb, depth

    def get_intrinsics(self):
        return self._camera_thread.camera.get_intrinsics()

    def get_projection_matrix(self):
        return self._camera_thread.camera.get_projection_matrix()

    def get_camera_matrix(self):
        return self._camera_thread.camera.get_camera_matrix()

    def get_dist_coeffs(self):
        return self._camera_thread.camera.get_dist_coeffs()

    def compute_pointcloud(self, depth_img):
        return self._camera_thread.camera.compute_pointcloud(depth_img)


class _CameraThread(threading.Thread):
    def __init__(self, camera_cfg):
        threading.Thread.__init__(self)
        self.camera = hydra.utils.instantiate(camera_cfg)
        self.daemon = True
        self.fps_controller = FpsController(camera_cfg.fps)
        self.flag_exit = False
        self.rgb = None
        self.depth = None

    def run(self):
        while not self.flag_exit:
            self.rgb, self.depth = self.camera.get_image()
            self.fps_controller.step()


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import cv2
    cfg = OmegaConf.load("../conf/cams/gripper_cam/framos.yaml")
    cam = ThreadedCamera(cfg)
    print(cam.get_intrinsics())
    while True:
        rgb, depth = cam.get_image()
        cv2.imshow("depth", depth)
        cv2.imshow("rgb", rgb[:, :, ::-1])
        cv2.waitKey(1)
