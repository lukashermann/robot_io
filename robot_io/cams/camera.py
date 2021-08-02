import cv2
import hydra.utils
import numpy as np


class Camera:
    def __init__(self):
        self.crop_coords = None
        self.resize_resolution = None

    def get_image(self):
        rgb, depth = self._get_image()
        rgb = self._crop_and_resize(rgb)
        depth = self._crop_and_resize(depth)
        return rgb, depth

    def _get_image(self):
        raise NotImplementedError

    def _crop_and_resize(self, img):
        if self.crop_coords is not None:
            c = self.crop_coords
            img = img[c[0]: c[1], c[2]:c[3]]
        if self.resize_resolution is not None:
            img = cv2.resize(img, tuple(self.resize_resolution))
        return img

    def get_intrinsics(self):
        raise NotImplementedError

    def get_projection_matrix(self):
        raise NotImplementedError

    def get_camera_matrix(self):
        raise NotImplementedError

    def get_dist_coeffs(self):
        raise NotImplementedError

    def compute_pointcloud(self, depth_img):
        raise NotImplementedError

    def project(self, X):
        if X.shape[0] == 3:
            if len(X.shape) == 1:
                X = np.append(X, 1)
            else:
                X = np.concatenate([X, np.ones((1, X.shape[1]))], axis=0)

        x = self.get_projection_matrix() @ X
        return x[0:2] / x[2]


def select_roi(cam):
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






