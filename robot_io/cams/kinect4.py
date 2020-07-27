import open3d as o3d
import numpy as np
import cv2


class Kinect4:
    def __init__(self, device=0, align_depth_to_color=True, config_path='config/default_config_kinect4.json'):
        if config_path is not None:
            config = o3d.io.read_azure_kinect_sensor_config(config_path)
        else:
            config = o3d.io.AzureKinectSensorConfig()
        self.flag_exit = False
        self.align_depth_to_color = align_depth_to_color
        self.sensor = o3d.io.AzureKinectSensor(config)

        if not self.sensor.connect(device):
            # raise RuntimeError('Failed to connect to sensor')
            pass
        data = np.load("config/kinect4_params.npz", allow_pickle=True)
        self.dist_coeffs = data['dist_coeffs']
        self.camera_matrix = data['camera_matrix']
        self.projection_matrix = data['projection_matrix']
        self.intrinsics = data['intrinsics'].item()

    def get_intrinsics(self):
        return self.intrinsics

    def get_projection_matrix(self):
        return self.projection_matrix

    def get_image(self, undistorted=False):
        rgbd = None
        while rgbd is None:
            rgbd = self.sensor.capture_frame(self.align_depth_to_color)
        rgb = np.asarray(rgbd.color)
        depth = (np.asarray(rgbd.depth)).astype('float') / 1000
        if undistorted:
            rgb = cv2.undistort(rgb, self.camera_matrix, self.dist_coeffs)
            depth = cv2.undistort(depth, self.camera_matrix, self.dist_coeffs)
        return rgb, depth


if __name__ == "__main__":
    kinect = Kinect4(config_path='config/default_config_kinect4.json')
    while 1:
        rgb, depth = kinect.get_image(undistorted=False)
        cv2.imshow("win", depth)
        cv2.imshow("win2", rgb[:,:,::-1])
        cv2.waitKey(1)

