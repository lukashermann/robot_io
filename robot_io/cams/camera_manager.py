import cv2
import hydra
import numpy as np

from robot_io.cams.threaded_camera import ThreadedCamera


class CameraManager:
    """
    Class for handling different cameras
    """
    def __init__(self, use_gripper_cam, use_static_cam, gripper_cam, static_cam, threaded_cameras):
        self.gripper_cam = None
        self.static_cam = None
        if use_gripper_cam:
            if threaded_cameras:
                self.gripper_cam = ThreadedCamera(gripper_cam)
            else:
                self.gripper_cam = hydra.utils.instantiate(gripper_cam)
        if use_static_cam:
            if threaded_cameras:
                self.static_cam = ThreadedCamera(static_cam)
            else:
                self.static_cam = hydra.utils.instantiate(static_cam)
        self.obs = None

    def get_images(self):
        obs = {}
        if self.gripper_cam is not None:
            rgb_gripper, depth_gripper = self.gripper_cam.get_image()
            obs['rgb_gripper'] = rgb_gripper
            obs['depth_gripper'] = depth_gripper
        if self.static_cam is not None:
            rgb, depth = self.static_cam.get_image()
            obs[f'rgb_static'] = rgb
            obs[f'depth_static'] = depth
        self.obs = obs
        return obs

    def save_calibration(self, robot_name):
        camera_info = {}
        if self.gripper_cam is not None:
            camera_info["gripper_extrinsic_calibration"] = self.gripper_cam.get_extrinsic_calibration(robot_name)
            camera_info["gripper_intrinsics"] = self.gripper_cam.get_intrinsics()
        if self.static_cam is not None:
            camera_info["static_extrinsic_calibration"] = self.static_cam.get_extrinsic_calibration(robot_name)
            camera_info["static_intrinsics"] = self.static_cam.get_intrinsics()
        np.savez("camera_info.npz", **camera_info)

    def normalize_depth(self, img):
        img_mask = img == 0
        istats = (np.min(img[img > 0]), np.max(img))
        imrange = (img.astype("float32") - istats[0]) / (istats[1] - istats[0])
        imrange[img_mask] = 0
        imrange = 255.0 * imrange
        imsz = imrange.shape
        nchan = 1
        if len(imsz) == 3:
            nchan = imsz[2]
        imgcanvas = np.zeros((imsz[0], imsz[1], nchan), dtype="uint8")
        imgcanvas[0: imsz[0], 0: imsz[1]] = imrange.reshape((imsz[0], imsz[1], nchan))
        return imgcanvas

    def render(self):
        if "rgb_gripper" in self.obs:
            cv2.imshow("rgb_gripper", self.obs["rgb_gripper"][:, :, ::-1])
        if "depth_gripper" in self.obs:
            depth_img_gripper = self.normalize_depth(self.obs["depth_gripper"])
            depth_img_gripper = cv2.applyColorMap(depth_img_gripper, cv2.COLORMAP_JET)
            cv2.imshow("depth_gripper", depth_img_gripper)
        if "rgb_static" in self.obs:
            cv2.imshow("rgb_static", self.obs["rgb_static"][:, :, ::-1])
        cv2.waitKey(1)
