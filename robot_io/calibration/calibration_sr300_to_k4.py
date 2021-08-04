

# from robot_io.cams.kinect4_threading import Kinect4

# q1: what is kinect4_threading doing in comparison to kinect4?
# A1: We are recording videos of each interaction and we need to maintain a fixed frame rate for the video.
# Therefore we use a thread for camera to make its frame process independent of other stuff.

# q2: what are the kinect4_params_1080p.npz for in comparision to config_kinect4_1080p.json?
# A2: in kinect4_params_1080p.npz we store kinect4 parameters like camera intrinsics and camera distortion coefficients. These parameters are camera specific.
# With config_kinect4_1080p.json we can configure the output of camera as we wish, for example we can set its frame rate or resolution.
#robot_io.cams.

# q3: why can does it fail to open device here, but it works inside kinect4.py?
# A3: you have already opened device above
# k4 = Kinect4(config_path='config/config_kinect4_1080p.json')

# q5: hardware reset problem for you?
# A5: no, in fact I assume that we will find below parameters (by hand or with optimization) and always start camera with these options.

# q6: why is image taken with this option so much different from realsenseviewer?
# A6: difference between RealsenseSR300 and RealsenseSR300Thread


import open3d as o3d
import numpy as np
import cv2
import ctypes

from robot_io.cams.kinect4.kinect4_threading import Kinect4


from robot_io.cams.realsense.realsenseSR300_threading import RealsenseSR300
# from realsenseSR300_threading import RealsenseSR300Thread
import pyrealsense as pyrs
from pyrealsense.constants import rs_option

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from scipy.spatial.transform import Rotation as R
from robot_io.kuka_iiwa.iiwa_controller import IIWAController

import time
import matplotlib.pyplot as plt

# note: threading makes big difference for sr300 rgb images
# sr300 = RealsenseSR300Thread(img_type='rgb_depth')
# k4 = Kinect4(config_path='config/config_kinect4_1080p.json')


# # q7: depth is aligned to rgb?
# sr300_intrinsics = {
#     'fx': 617.8902587890625,
#     'fy': 617.8903198242188,
#     'cx': 315.20367431640625,
#     'cy': 245.70614624023438,
#     'width': 640,
#     'height': 480
#
# }
#
# k4_intrinsics = {
#     'cx': 956.4253540039062,
#     'cy': 553.5712280273438,
#     'fx': 917.6631469726562,
#     'fy': 917.4476318359375,
#     'width': 1920,
#     'height': 1080
# }

'''
sr300_intrinsics_all = {
    'rgb': {
        'width': 640,
        'height': 480,
        'fx': 617.8902587890625,
        'fy': 617.8903198242188,
        'cx': 315.20367431640625,
        'cy': 245.70614624023438
    },
    'depth': {
        'width': 640,
        'height': 480,
        'fx': 474.3828125,
        'fy': 474.3828430175781,
        'cx': 317.2261962890625,
        'cy': 245.13381958007812
    }
}
'''

############## SR300 Params + Ranges ################
# white_balance  from: 2800 to: 6500  def: 3400
# exposure       from: 39   to: 10000 def: 300
# brightness     from: -64  to: 64    def: 50
# contrast       from: 0    to: 100   def: 50
# saturation     from: 0    to: 100   def: 64
# sharpness      from: 0    to: 100   def: 50
# gain           from: 0    to: 128   def: 64

# gamma          from: 100  to: 500   def: 300
# hue            from: -180 to: 180   def: 0
#
config_space = [
    Real(4094.0, 4094.1, name='white_balance'),
    # Real(2800.0, 6500.0, name='white_balance'),
    Real(637.0, 637.1, name='exposure'),
    # Real(39.0, 10000.0, name='exposure'),
    Real(30.0, 30.1, name='brightness'),
    # Real(-64.0, 64.0, name='brightness'),
    Real(34.0, 34.1, name='contrast'),
    # Real(0.0, 100.0, name='contrast'),
    Real(77.0, 77.1, name='saturation'),
    # Real(0.0, 100.0, name='saturation'),
    Real(50.0, 50.1, name='sharpness'),
    # Real(0.0, 100.0, name='sharpness'),
    Real(64.0, 64.1, name='gain')
    # Real(0.0, 128.0, name='gain')
    #Integer(0, 255, name='offset_red'),
    #Integer(0, 255, name='offset_green'),
    #Integer(0, 255, name='offset_blue')
]
#
# sr300_options_def = [(rs_option.RS_OPTION_COLOR_ENABLE_AUTO_WHITE_BALANCE, 0),
#                   (rs_option.RS_OPTION_COLOR_WHITE_BALANCE, 3400.0),
#                   (rs_option.RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE, 0),
#                   (rs_option.RS_OPTION_COLOR_EXPOSURE, 300.0),
#                   (rs_option.RS_OPTION_COLOR_BRIGHTNESS, 50.0),
#                   (rs_option.RS_OPTION_COLOR_CONTRAST, 50.0),
#                   (rs_option.RS_OPTION_COLOR_SATURATION, 64.0),
#                   (rs_option.RS_OPTION_COLOR_SHARPNESS, 50.0),
#                   (rs_option.RS_OPTION_COLOR_GAIN, 64.0),
#                   (rs_option.RS_OPTION_COLOR_GAMMA, 300.0),
#                   (rs_option.RS_OPTION_COLOR_HUE, 0.0)]
#
# sr300.cam.set_device_options(*zip(*sr300_options_def))
#
#
# sr300_start_x = sr300_start_y = sr300_dim_x = sr300_dim_y = 0
# k4_start_x = k4_start_y = k4_dim_x = k4_dim_y = 0

def set_sr300_device_options(params):
    sr300_options = [(rs_option.RS_OPTION_COLOR_ENABLE_AUTO_WHITE_BALANCE, 0),
                      (rs_option.RS_OPTION_COLOR_WHITE_BALANCE, params['white_balance']),
                      (rs_option.RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE, 0),
                      (rs_option.RS_OPTION_COLOR_EXPOSURE, params['exposure']),
                      (rs_option.RS_OPTION_COLOR_BRIGHTNESS, params['brightness']),
                      (rs_option.RS_OPTION_COLOR_CONTRAST, params['contrast']),
                      (rs_option.RS_OPTION_COLOR_SATURATION, params['saturation']),
                      (rs_option.RS_OPTION_COLOR_SHARPNESS, params['sharpness']),
                      (rs_option.RS_OPTION_COLOR_GAIN, params['gain']),
                      (rs_option.RS_OPTION_COLOR_GAMMA, 300.0),
                      (rs_option.RS_OPTION_COLOR_HUE, 0.0)]

    sr300.set_device_options(sr300_options)


def project_pixel_to_point(u, v, z, intrinsics):
    x = (u - intrinsics['cx']) * z / intrinsics['fx']
    y = (v - intrinsics['cy']) * z / intrinsics['fy']
    return np.array([x, y, z, 1])


def rgbd_to_xyz1rgb(rgb, depth, intrinsics):
    dim_y = rgb.shape[0]
    dim_x = rgb.shape[1]

    u, v = np.meshgrid(range(dim_x), range(dim_y))


    z = depth
    x = np.multiply(u - intrinsics['cx'], z) / intrinsics['fx']
    y = np.multiply(v - intrinsics['cy'], z) / intrinsics['fy']

    xyz1rgb = np.stack((
        x.flatten(),
        y.flatten(),
        z.flatten(),
        np.ones(z.shape).flatten(),
        rgb[:, :, 0].flatten(),
        rgb[:, :, 1].flatten(),
        rgb[:, :, 2].flatten()
    ), axis=1)

    # filtering depth = 0.0
    xyz1rgb = xyz1rgb[xyz1rgb[:, 2] > 0.0]
    return xyz1rgb

def xyz1_transform(xyz1, transf):
    return np.dot(transf, xyz1.T).T

def xyz1rgb_to_rgb(xyz1rgb, intrinsics):

    u = np.divide(xyz1rgb[:, 0], xyz1rgb[:, 2]) * intrinsics['fx'] + intrinsics['cx']
    v = np.divide(xyz1rgb[:, 1], xyz1rgb[:, 2]) * intrinsics['fy'] + intrinsics['cy']

    u_compl = u.astype(int)
    v_compl = v.astype(int)

    u = u_compl[(u_compl >= 0) & (u_compl < intrinsics['width']) & (v_compl >= 0) & (v_compl < intrinsics['height'])]
    v = v_compl[(u_compl >= 0) & (u_compl < intrinsics['width']) & (v_compl >= 0) & (v_compl < intrinsics['height'])]
    xyz1rgb = xyz1rgb[(u_compl >= 0) & (u_compl < intrinsics['width']) & (v_compl >= 0) & (v_compl < intrinsics['height'])]


    rgb = np.zeros((intrinsics['height'], intrinsics['width'], 3), dtype=np.uint8)
    mask = np.zeros((intrinsics['height'], intrinsics['width']), dtype=np.bool)

    print("len(v)", len(v))
    for i in range(len(v)):
        rgb[v[i], u[i], :] = xyz1rgb[i, 4:].astype(int)
    #rgb[v[:], u[:], :] = xyz1rgb[:, 4:].astype(int)

    mask[v[:], u[:]] = 1

    #cv2.imshow('rgb', rgb[:, :, ::-1])
    #cv2.imshow('mask', mask * 255)
    #cv2.waitKey(0)

    return mask, rgb


# def project_point_to_pixel(x, y, z, intrinsics):
#     u = x * intrinsics['fx'] / z + intrinsics['cx']
#     v = y * intrinsics['fy'] / z + intrinsics['cy']
#     return np.array([u, v])
#
#
# #  1. retrieve rgbd
# rgb_k4, depth_k4 = k4.get_image(undistorted=True)
#
# #  2. project to rgbd to 3d space
# xyz1rgb_k4 = rgbd_to_xyz1rgb(rgb_k4, depth_k4, k4_intrinsics)
#
# #  3. transform from k4 coords to sr300 coords
# data = np.load('/home/kuka/lang/robot/iman/robot_pushing/robot_pushing/calibration/calibration.npz')
# T_cam_robot = data['T_cam_robot']
# T_grippercam_tcp = data['T_grippercam_tcp']
#
# robot = IIWAController(use_impedance=False, joint_vel=0.1, joint_acc=0.3, gripper_rot_vel=0.3, workspace_limits=(0, 2, -1, 1, 0.1, 2))
# time.sleep(1)
# curr_pos = robot.get_joint_info()
# print("curr_pos", curr_pos)
# trans = curr_pos[0:3]
# rot = R.from_euler('xyz', curr_pos[3:6], degrees=False)
# T_R_EE = np.eye(4)
# T_R_EE[:3, 3] = trans
# T_R_EE[:3, :3] = rot.as_dcm()
# print("T_R_EE", T_R_EE)
#
# transf_k4_to_sr300 = np.eye(4)
# xyz1rgb_sr300 = xyz1rgb_k4.copy()
# xyz1rgb_sr300[:, :4] = xyz1_transform(xyz1rgb_k4[:, :4], transf_k4_to_sr300)
#
# #  4. project 3d points to pixels from sr300 plane
# mask_sr300_warped, rgb_sr300_warped = xyz1rgb_to_rgb(xyz1rgb_sr300, k4_intrinsics)



'''

np.savez("calibration/calibration.npz", T_robot_arena=self.T_robot_arena,
                 T_cam_robot=self.T_cam_robot,
                 T_cam_arena=self.T_cam_arena,
                 # T_depth_rgb=self.T_depth_rgb,
                 arena_center_xy_rgb=self.arena_center_xy_rgb,
                 T_grippercam_tcp=self.T_grippercam_tcp)
# q8: what is tcp? 
'''
@use_named_args(config_space)
def objective(**params):
    set_sr300_device_options(params)
    time.sleep(0.1)
    # sr300_rgb: numpy.ndarray with 480 x 640 x 3 uint8
    #sr300_rgb, depth = sr300.get_image()

    # k4_rgb: numpy.ndarray with 1080 x 1920 x 3
    #k4_rgb, depth = k4.get_image(undistorted=True)

    rgb_k4, depth_k4 = k4.get_image(undistorted=True)

    sr300_mask_warped, sr300_rgb_warped = get_sr300_rgb_warped(rgb_k4,
                                                               depth_k4,
                                                               T_k4_sr300,
                                                               k4_intrinsics,
                                                               sr300_intrinsics)

    sr300_rgb, sr300_depth, _ = sr300.get_image()

    sr300_rgb_patch = sr300_rgb[sr300_start_y:sr300_start_y + sr300_dim_y,
                                sr300_start_x:sr300_start_x + sr300_dim_x, :]

    sr300_rgb_warped_patch = sr300_rgb_warped[sr300_start_y:sr300_start_y + sr300_dim_y,
                      sr300_start_x:sr300_start_x + sr300_dim_x, :]

    sr300_mask_warped_patch = sr300_mask_warped[sr300_start_y:sr300_start_y + sr300_dim_y,
                             sr300_start_x:sr300_start_x + sr300_dim_x]

    #k4_rgb_patch = k4_rgb[k4_start_y:k4_start_y + k4_dim_y,
    #                      k4_start_x:k4_start_x + k4_dim_x, :]
    #k4_rgb_patch = cv2.resize(k4_rgb_patch, sr300_rgb_patch.shape[-2::-1])

    #rgbs = np.concatenate((sr300_rgb_patch, sr300_rgb_warped_patch), axis=1)
    #cv2.imshow("rgbs", rgbs[:, :, ::-1])
    #time.sleep(0.1)
    #cv2.waitKey(0)

    rgb_patches_diff = sr300_rgb_patch.astype(np.int) - sr300_rgb_warped_patch.astype(np.int)
    rgb_patches_diff[~sr300_mask_warped_patch] = 0

    offset_red = np.mean(rgb_patches_diff[:, :, 0]).astype(np.int)
    offset_green = np.mean(rgb_patches_diff[:, :, 1]).astype(np.int)
    offset_blue = np.mean(rgb_patches_diff[:, :, 2]).astype(np.int)

    #rgb_patches_diff[:, :, 0] -= offset_red
    #rgb_patches_diff[:, :, 1] -= offset_green
    #rgb_patches_diff[:, :, 2] -= offset_blue

    print('offsets rgb: ', offset_red, offset_green, offset_blue)

    loss = np.mean(np.abs(rgb_patches_diff.flatten()))
    #print(params)
    #print('loss: ', loss)
    return loss

def selectROI_sr300():
    # sr300_rgb: numpy.ndarray with 480 x 640 x 3
    sr300_rgb, depth, _ = sr300.get_image()
    print(depth.shape)
    # cv2.selectROI(...) returns tuple
    return cv2.selectROI(sr300_rgb[:, :, ::-1])

def selectROI_k4():
    # k4_rgb: numpy.ndarray with 480 x 640 x 3
    k4_rgb, depth = k4.get_image(undistorted=True)
    print(depth.shape)
    # cv2.selectROI(...) returns tuple
    return cv2.selectROI(k4_rgb[:, :, ::-1])

def find_opt_sr300_options(n_random_starts= 10, n_calls = 50):
    opt_results = gp_minimize(objective, config_space,
                              n_calls=n_calls,
                              n_random_starts = n_random_starts,
                              random_state=None,
                              verbose=True)

    opt_loss = opt_results.fun
    opt_params = {
        'white_balance': opt_results.x[0],
        'exposure': opt_results.x[1],
        'brightness': opt_results.x[2],
        'contrast': opt_results.x[3],
        'saturation': opt_results.x[4],
        'sharpness': opt_results.x[5],
        'gain': opt_results.x[6]
    }

    print('optimal params \n', opt_params)
    print('optimal loss \n', opt_loss)

    return opt_params


def show_rgbs(sr300_options):
    set_sr300_device_options(sr300_options)

    # sr300_rgb: numpy.ndarray with 480 x 640 x 3
    # depth aligned to rgb
    sr300_rgb, depth,_ = sr300.get_image()


    # k4_rgb: numpy.ndarray with 1080 x 1920 x 3
    # depth aligned to rgb
    k4_rgb, depth = k4.get_image(undistorted=True)
    k4_rgb = cv2.resize(k4_rgb, sr300_rgb.shape[-2::-1])


    rgbs = np.concatenate((sr300_rgb, k4_rgb), axis=1)
    cv2.imshow("rgbs", rgbs[:,:,::-1])
    cv2.waitKey(0)


def visualize_patch_diff(sr300_options):
    set_sr300_device_options(sr300_options)
    time.sleep(0.5)
    # sr300_rgb: numpy.ndarray with 480 x 640 x 3
    # sr300_rgb, depth = sr300.get_image()

    # k4_rgb: numpy.ndarray with 1080 x 1920 x 3
    # k4_rgb, depth = k4.get_image(undistorted=True)

    rgb_k4, depth_k4 = k4.get_image(undistorted=True)

    sr300_mask_warped, sr300_rgb_warped = get_sr300_rgb_warped(rgb_k4,
                                                               depth_k4,
                                                               T_k4_sr300,
                                                               k4_intrinsics,
                                                               sr300_intrinsics)

    sr300_rgb, sr300_depth, _ = sr300.get_image()

    sr300_rgb_patch = sr300_rgb[sr300_start_y:sr300_start_y + sr300_dim_y,
                      sr300_start_x:sr300_start_x + sr300_dim_x, :]


    sr300_rgb_warped_patch = sr300_rgb_warped[sr300_start_y:sr300_start_y + sr300_dim_y,
                             sr300_start_x:sr300_start_x + sr300_dim_x, :]



    sr300_mask_warped_patch = sr300_mask_warped[sr300_start_y:sr300_start_y + sr300_dim_y,
                              sr300_start_x:sr300_start_x + sr300_dim_x]


    # k4_rgb_patch = k4_rgb[k4_start_y:k4_start_y + k4_dim_y,
    #                      k4_start_x:k4_start_x + k4_dim_x, :]
    # k4_rgb_patch = cv2.resize(k4_rgb_patch, sr300_rgb_patch.shape[-2::-1])

    rgb_patches_diff = sr300_rgb_patch.astype(np.int) - sr300_rgb_warped_patch.astype(np.int)
    rgb_patches_diff[~sr300_mask_warped_patch] = 0

    offset_red = np.mean(rgb_patches_diff[:, :, 0])
    offset_green = np.mean(rgb_patches_diff[:, :, 1])
    offset_blue = np.mean(rgb_patches_diff[:, :, 2])

    print('offsets rgb: ', offset_red, offset_green, offset_blue)

    sr300_rgb_patch[:, :, 0] -= offset_red.astype(np.uint8)
    sr300_rgb_patch[:, :, 1] -= offset_green.astype(np.uint8)
    sr300_rgb_patch[:, :, 2] -= offset_blue.astype(np.uint8)

    rgb_patches_diff = sr300_rgb_patch.astype(np.int) - sr300_rgb_warped_patch.astype(np.int)
    rgb_patches_diff[~sr300_mask_warped_patch] = 0

    print('err', np.mean(rgb_patches_diff.flatten()))

    rgbs = np.concatenate((sr300_rgb_patch, sr300_rgb_warped_patch), axis=1)

    cv2.imshow("rgbs", rgbs[:, :, ::-1])
    # time.sleep(0.1)
    cv2.waitKey(0)

    cv2.imshow("rgb_diff", rgb_patches_diff[:, :, ::-1].astype(np.uint8))
    # time.sleep(0.1)
    cv2.waitKey(0)




def get_sr300_rgb_warped(rgb_k4, depth_k4, T_k4_sr300, k4_intrinsics, sr300_intrinsics):

    #  1. project to rgbd to 3d space
    xyz1rgb_k4 = rgbd_to_xyz1rgb(rgb_k4, depth_k4, k4_intrinsics)

    #  2. transform from k4 coords to sr300 coords
    xyz1rgb_sr300 = xyz1rgb_k4.copy()
    xyz1rgb_sr300[:, :4] = xyz1_transform(xyz1rgb_k4[:, :4], np.linalg.inv(T_k4_sr300))

    #  3. project 3d points to pixels from sr300 plane
    mask_sr300_warped, rgb_sr300_warped = xyz1rgb_to_rgb(xyz1rgb_sr300, sr300_intrinsics)

    return mask_sr300_warped, rgb_sr300_warped

if __name__ == "__main__":
    sr300 = RealsenseSR300(img_type='rgb_depth')
    k4 = Kinect4()

    # q7: depth is aligned to rgb?
    sr300_intrinsics = {
        'fx': 617.8902587890625,
        'fy': 617.8903198242188,
        'cx': 315.20367431640625,
        'cy': 245.70614624023438,
        'width': 640,
        'height': 480

    }

    k4_intrinsics = {
        'cx': 956.4253540039062,
        'cy': 553.5712280273438,
        'fx': 917.6631469726562,
        'fy': 917.4476318359375,
        'width': 1920,
        'height': 1080
    }

    opt_params = {
        'white_balance': 4985.863918777444,
        'exposure': 4637.480240257885,
        'brightness': 58.84815853652762,
        'contrast': 30.709146596228486,
        'saturation': 68.75703580142678,
        'sharpness': 50.03586888843052,
        'gain': 64.07292376692126
    }

    opt_params = {
        'white_balance': 4102.891497425365,
        'exposure': 3000.0,
        'brightness': 30.0,
        'contrast': 11.286314030105814,
        'saturation': 64.73741929775156,
        'sharpness': 50.00588375617516,
        'gain': 64.1
    }

    opt_params = {
        'white_balance': 3000.0,
        'exposure': 1681.4417372072705,
        'brightness': 34.938971295696575,
        'contrast': 10.0,
        'saturation': 75.04295731463128,
        'sharpness': 50.083587538414335,
        'gain': 64.0793237627221
    }
    '''
    opt_params = {
        'white_balance': 4269.6746800717865,
        'exposure': 300.0,
        'brightness': 30.813620556481172,
        'contrast': 50.0,
        'saturation': 62.568043011647596,
        'sharpness': 50.02254785823696,
        'gain': 64.07832057561541
    }
    '''
    opt_params = {
        'white_balance': 4094.5609342299676,
        'exposure': 637.0694099222924,
        'brightness': 30.0,
        'contrast': 34.22499590933867,
        'saturation': 77.41023389719855,
        'sharpness': 50.071698364875736,
        'gain': 64.01785157407843,
        'offset_red': 4,
        'offset_green': 0,
        'offset_blue': 4
    }
    '''
    {'white_balance': 4289.404485702248, 'exposure': 898.3062973755027, 'brightness': -1.3006271327579455,
     'contrast': 53.27723098007704, 'saturation': 84.25073689209316, 'sharpness': 50.01133861612505,
     'gain': 64.07535135224518}
    '''


    config_space = [
        Real(3000.0, 5000.0, name='white_balance'),
        # Real(2800.0, 6500.0, name='white_balance'),
        Real(390.0, 5000.0, name='exposure'),
        # Real(39.0, 10000.0, name='exposure'),
        Real(-64.0, 64.0, name='brightness'),
        # Real(-64.0, 64.0, name='brightness'),
        Real(30.0, 90.0, name='contrast'),
        # Real(0.0, 100.0, name='contrast'),
        Real(30.0, 90.0, name='saturation'),
        # Real(0.0, 100.0, name='saturation'),
        Real(50.0, 50.1, name='sharpness'),
        # Real(0.0, 100.0, name='sharpness'),
        Real(64.0, 64.1, name='gain')
        # Real(0.0, 128.0, name='gain'),
        #Integer(0, 255, name='offset_red'),
        #Integer(0, 255, name='offset_green'),
        #Integer(0, 255, name='offset_blue')
    ]

    sr300_options_def = [(rs_option.RS_OPTION_COLOR_ENABLE_AUTO_WHITE_BALANCE, 0),
                         (rs_option.RS_OPTION_COLOR_WHITE_BALANCE, 3400.0),
                         (rs_option.RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE, 0),
                         (rs_option.RS_OPTION_COLOR_EXPOSURE, 300.0),
                         (rs_option.RS_OPTION_COLOR_BRIGHTNESS, 50.0),
                         (rs_option.RS_OPTION_COLOR_CONTRAST, 50.0),
                         (rs_option.RS_OPTION_COLOR_SATURATION, 64.0),
                         (rs_option.RS_OPTION_COLOR_SHARPNESS, 50.0),
                         (rs_option.RS_OPTION_COLOR_GAIN, 64.0),
                         (rs_option.RS_OPTION_COLOR_GAMMA, 300.0),
                         (rs_option.RS_OPTION_COLOR_HUE, 0.0)]

    sr300.set_device_options(sr300_options_def)

    sr300_start_x = sr300_start_y = sr300_dim_x = sr300_dim_y = 0
    k4_start_x = k4_start_y = k4_dim_x = k4_dim_y = 0

    data = np.load('/home/kuka/lang/robot/iman/robot_pushing/robot_pushing/calibration/calibration.npz')
    T_k4_robot = data['T_cam_robot']
    T_sr300_ee = data['T_grippercam_tcp']

    robot = IIWAController(use_impedance=False, joint_vel=0.1, joint_acc=0.3, gripper_rot_vel=0.3,
                           workspace_limits=(0, 2, -1, 1, 0.1, 2))
    time.sleep(0.1)
    curr_pos = robot.get_tcp_pose()
    trans = curr_pos[0:3]
    rot = R.from_euler('xyz', curr_pos[3:6], degrees=False)
    T_robot_ee = np.eye(4)
    T_robot_ee[:3, 3] = trans
    T_robot_ee[:3, :3] = rot.as_dcm()

    T_k4_sr300 = T_k4_robot @ T_robot_ee @ np.linalg.inv(T_sr300_ee)


    sr300_start_x, sr300_start_y, sr300_dim_x, sr300_dim_y = selectROI_sr300()
    #k4_start_x, k4_start_y, k4_dim_x, k4_dim_y = selectROI_k4()
    opt_params = find_opt_sr300_options(n_random_starts=60, n_calls=100)

    print(opt_params)
    visualize_patch_diff(opt_params)
    #show_rgbs(opt_params)




