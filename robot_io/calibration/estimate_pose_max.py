import argparse, os, glob
import numpy as np
import imageio as ioi
from tqdm import tqdm

from tag_detector.utils.general_util import json_load, json_dump, my_mkdir
from tag_detector.core.BoardDetector import BoardDetector
from tag_detector.core.TagPoseEstimator import TagPoseEstimator


def parse_calib(data_path):
    calib_path1 = os.path.join(data_path, 'info.json')
    if os.path.isfile(calib_path1):
        calib = json_load(calib_path1)["camera"]["calibration"]
        cam_intrinsic = np.eye(3)
        cam_intrinsic[0, 0] = calib["fx"]
        cam_intrinsic[1, 1] = calib["fy"]
        cam_intrinsic[0, 2] = calib["ppx"]
        cam_intrinsic[1, 2] = calib["ppy"]
        cam_dist = np.array([0., 0., 0., 0., 0.])
        return cam_intrinsic, cam_dist
    else:
        calib_path2 = os.path.join(data_path, '../calib.json')
        assert os.path.exists(calib_path2), 'Calibfile not found.'
        calib = json_load(calib_path2)
        cam_intrinsic, cam_dist = np.array(calib['K']), np.array(calib['dist'])
        print("Warning Deprecated calibration file format.")

    return cam_intrinsic, cam_dist

def parse_folder(data_path):
    # calibration
    cam_intrinsic, cam_dist = parse_calib(data_path)
    print(cam_intrinsic, cam_dist)

    # find images
    i = 1
    img_list, depth_list, idx_list = list(), list(), list()
    while True:
        p1 = os.path.join(data_path, 'rgb_%04d.png' % i)
        p2 = os.path.join(data_path, 'depth_%04d.png' % i)

        if not os.path.exists(p1) or not os.path.exists(p2):
            break

        img_list.append(p1)
        depth_list.append(p2)
        idx_list.append(i)
        i += 1

    assert all([os.path.exists(x) for x in img_list]), 'Image not found.'
    assert all([os.path.exists(x) for x in depth_list]), 'Depth map not found.'
    assert len(depth_list) == len(img_list), 'There need to be equally many'

    return idx_list, img_list, depth_list, cam_intrinsic, cam_dist

def estimate_pose(est, K, dist, p2d, pid, thresh=4):
    if p2d.shape[0] < thresh:
        return None

    ret = est.estimate_relative_cam_pose(K, dist, p2d, pid)
    if ret is None:
        return None
    points3d_pred, R, t_rel = ret
    M_w2c = np.concatenate([R, t_rel], 1)
    M_w2c = np.concatenate([M_w2c, np.array([[0.0, 0.0, 0.0, 1.0]])], 0)
    return M_w2c


if __name__ == '__main__':
    """
        Estimates the camera location relative to the marker board. 
    """

    parser = argparse.ArgumentParser(description='Record data with a Realsense camera.')
    parser.add_argument('marker', type=str, help='Marker description file.')
    parser.add_argument('data_path', type=str, help='Path to where the recorded data is.')
    parser.add_argument('--min_tags', type=int, default=3, help='Minimal number of Apriltags visible.')
    args = parser.parse_args()

    idx_list, img_list, depth_list, K, dist = parse_folder(args.data_path)
    output_path_pose = os.path.join(args.data_path, 'pose', '%08d.json')
    my_mkdir(output_path_pose, is_file=True)

    # set up detector and estimator
    detector = BoardDetector(args.marker)
    estimator = TagPoseEstimator(detector)
    
    for img in img_list[0:1]:
        # Test detection which loads file from filename
        points2d, point_ids = detector.process_image(img)
        M_w2c = estimate_pose(estimator, K, dist, points2d, point_ids,
                              thresh=args.min_tags*4)
        print(M_w2c)

        # Test detection which uses numpy array
        from PIL import Image
        image_np = np.asarray(Image.open(img))
        points2d, point_ids = detector.process_image_m(image_np)
        M_w2c_2 = estimate_pose(estimator, K, dist, points2d, point_ids,
                              thresh=args.min_tags*4)
        print(M_w2c_2)
        if M_w2c is None:
            print('Failed: ', img)
            continue

        # save camera pose
        # json_dump(output_path_pose % i, M_w2c)

    # Test detection wich loads files in batch mode.
    """
    points2d, point_ids = detector.process_image_batch(img_list)

    # iterate frames
    for i, p2d, pid, img, depth in tqdm(zip(idx_list, points2d, point_ids, img_list, depth_list),
                                        total=len(idx_list), desc='Estimating pose'):
        M_w2c = estimate_pose(estimator,
                              K, dist, p2d, pid,
                              thresh=args.min_tags*4)
        if M_w2c is None:
            print('Failed: ', img)
            continue

        # save camera pose
        json_dump(output_path_pose % i, M_w2c)
   """
