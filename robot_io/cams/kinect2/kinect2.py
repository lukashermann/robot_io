import numpy as np
import cv2
import cv2.aruco as aruco
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
import time


class Kinect2:
    def __init__(self, img_type="rgb"):
        assert img_type in ['rgb', "rgb_depth"]
        try:
            from pylibfreenect2 import OpenGLPacketPipeline
            pipeline = OpenGLPacketPipeline()
        except:
            try:
                from pylibfreenect2 import OpenCLPacketPipeline
                pipeline = OpenCLPacketPipeline()
            except:
                from pylibfreenect2 import CpuPacketPipeline
                pipeline = CpuPacketPipeline()
        print("Packet pipeline:", type(pipeline).__name__)

        # Create and set logger
        self.logger = createConsoleLogger(LoggerLevel.Error)
        setGlobalLogger(self.logger)

        self.fn = Freenect2()
        num_devices = self.fn.enumerateDevices()
        if num_devices == 0:
            print("No device connected!")
            sys.exit(1)

        serial = self.fn.getDeviceSerialNumber(0)
        self.device = self.fn.openDevice(serial, pipeline=pipeline)

        self.listener = SyncMultiFrameListener(
            FrameType.Color)
        self.img_type = img_type
        # Register listeners
        if img_type == 'rgb':
            self.listener = SyncMultiFrameListener(
                FrameType.Color)
            self.device.setColorFrameListener(self.listener)
            self.device.startStreams(rgb=True, depth=False)
            params = self.get_intrinsics()
            self.camera_matrix_rgb = np.array([[params['rgb']['fx'], 0, params['rgb']['cx']],
                                              [0, params['rgb']['fy'], params['rgb']['cy']],
                                              [0, 0, 1]])
        else:
            self.listener = SyncMultiFrameListener(
                FrameType.Color | FrameType.Ir | FrameType.Depth)
            self.device.setColorFrameListener(self.listener)
            self.device.setIrAndDepthFrameListener(self.listener)
            self.device.start()

            self.registration = Registration(self.device.getIrCameraParams(),
                                             self.device.getColorCameraParams())

            self.undistorted = Frame(512, 424, 4)
            self.registered = Frame(512, 424, 4)

            # Optinal parameters for registration
            # set True if you need
            need_bigdepth = True
            need_color_depth_map = True

            self.bigdepth = Frame(1920, 1082, 4) if need_bigdepth else None
            self.color_depth_map = np.zeros((424, 512),  np.int32).ravel() \
                if need_color_depth_map else None

            params = self.get_intrinsics()
            self.camera_matrix_rgb = np.array([[params['rgb']['fx'], 0, params['rgb']['cx']],
                                               [0, params['rgb']['fy'], params['rgb']['cy']],
                                               [0, 0, 1]])
            self.camera_matrix_depth = np.array([[params['depth']['fx'], 0, params['depth']['cx']],
                                           [0, params['depth']['fy'], params['depth']['cy']],
                                           [0, 0, 1]])

    def __del__(self):
        self.device.stop()
        self.device.close()

    def get_image(self):
        frames = self.listener.waitForNewFrame()

        color = frames["color"]
        if self.img_type == 'rgb':
            color = np.array(color.asarray())[:,::-1, :3]
            self.listener.release(frames)
            return color, None, None
        else:
            ir = frames["ir"]
            depth = frames["depth"]

            self.registration.apply(color, depth, self.undistorted, self.registered,
                                    bigdepth=self.bigdepth,
                                    color_depth_map=self.color_depth_map)

            # print(self.registration.getPointXYZRGB(self.undistorted, self.registered, 200,200))

            color = np.array(color.asarray())[:, ::-1, :3]

            # NOTE for visualization:
            # cv2.imshow without OpenGL backend seems to be quite slow to draw all
            # things below. Try commenting out some imshow if you don't have a fast
            # visualization backend.
            # cv2.imshow("ir", ir.asarray() / 65535.)
            # cv2.imshow("depth", depth.asarray() / 4500.)
            # cv2.imshow("color", cv2.resize(color.asarray(),
            #                                (int(1920 / 3), int(1080 / 3))))
            # cv2.imshow("registered", self.registered.asarray(np.uint8))
            #
            # if need_bigdepth:
            #     cv2.imshow("bigdepth", cv2.resize(bigdepth.asarray(np.float32),
            #                                       (int(1920 / 3), int(1082 / 3))))
            # if need_color_depth_map:
            #     cv2.imshow("color_depth_map", color_depth_map.reshape(424, 512))

            self.listener.release(frames)
            return color, self.registered.asarray(np.uint8)[:, ::-1, :3], depth.asarray()[:, ::-1] / 1000

    def get_intrinsics(self):
        params = self.device.getColorCameraParams()
        params_dict = {'rgb':{'cx': params.cx, 'cy': params.cy,'fx': params.fx,'fy': params.fy, 'width': 1920, 'height': 1080}}
        if self.img_type != 'rgb':
            params_ir = self.device.getIrCameraParams()
            params_dict['depth'] = {'cx': params_ir.cx, 'cy': params_ir.cy,'fx': params_ir.fx,'fy': params_ir.fy, 'width': 512, 'height': 424}
        return params_dict


if __name__ == "__main__":
    kinect = Kinect2(img_type='rgb_depth')
    print(kinect.get_intrinsics())
    while 1:
        rgb, reg, depth = kinect.get_image()
        cv2.imshow("reg", reg)
        cv2.imshow("depth", depth)

        cv2.imshow("rgb", cv2.resize(rgb, (int(1920 / 3), int(1080 / 3))))
        cv2.waitKey(1)

