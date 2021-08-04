import numpy as np
import cv2


class RelativeCameraPoseEstimator(object):
    """ Estimates the relative pose of two cameras using the essential matrix.
        Needs at least 5 points that are not all coplanar.
        """
    def __init__(self, translation_scale, verbose=False):
        self.translation_scale = translation_scale
        self.verbose = verbose

    def estimate_pose(self, camera_intrinsic0, camera_intrinsic1,
                      points2d_cam0, points2d_cam1,
                      camera_dist0=None, camera_dist1=None,
                      use_ransac=False):
        """ Estimates the relative camera pose between two cameras from some given point correspondences. """
        # Check inputs
        assert len(camera_intrinsic0.shape) == 2, "camera_intrinsic shape mismatch. Should be (3,3)"
        assert camera_intrinsic0.shape[0] == 3, "camera_intrinsic shape mismatch. Should be (3,3)"
        assert camera_intrinsic0.shape[1] == 3, "camera_intrinsic shape mismatch. Should be (3,3)"
        assert len(camera_intrinsic1.shape) == 2, "camera_intrinsic shape mismatch. Should be (3,3)"
        assert camera_intrinsic1.shape[0] == 3, "camera_intrinsic shape mismatch. Should be (3,3)"
        assert camera_intrinsic1.shape[1] == 3, "camera_intrinsic shape mismatch. Should be (3,3)"

        if camera_dist0 is not None:
            camera_dist0 = np.squeeze(np.array(camera_dist0))
            assert len(camera_dist0.shape) == 1, "camera_dist shape mismatch. Should be of length 4 or 5."
            assert (camera_dist0.shape[0] == 4) or (camera_dist0.shape[0] == 5), "camera_dist shape mismatch. Should be of length 4 or 5."

        if camera_dist1 is not None:
            camera_dist1 = np.squeeze(np.array(camera_dist1))
            assert len(camera_dist1.shape) == 1, "camera_dist shape mismatch. Should be of length 4 or 5."
            assert (camera_dist1.shape[0] == 4) or (camera_dist1.shape[0] == 5), "camera_dist shape mismatch. Should be of length 4 or 5."

        assert len(points2d_cam0.shape) == 2, "points2d_cam0 shape mismatch. Should be Nx2"
        assert points2d_cam0.shape[0] > 5, "points2d_cam0 shape mismatch. Should be Nx2, with N>5"
        assert points2d_cam0.shape[1] == 2, "points2d_cam0 shape mismatch. Should be Nx2"
        assert len(points2d_cam1.shape) == 2, "points2d_cam1 shape mismatch. Should be Nx2"
        assert points2d_cam1.shape[0] > 5, "points2d_cam1 shape mismatch. Should be Nx2, with N>5"
        assert points2d_cam1.shape[1] == 2, "points2d_cam1 shape mismatch. Should be Nx2"
        assert points2d_cam0.shape[0] == points2d_cam1.shape[0], "points2d_cam0 and points2d_cam1 shape mismatch. They must contain an equal amount of points. "

        if camera_dist0 is None:
            camera_dist0 = np.zeros((1, 5))

        if camera_dist1 is None:
            camera_dist1 = np.zeros((1, 5))

        # normalize and undistort points: cam0
        points2d_cam0_norm = cv2.undistortPoints(np.expand_dims(points2d_cam0, 0), camera_intrinsic0, camera_dist0)
        points2d_cam0_norm = np.squeeze(points2d_cam0_norm)
        points2d_cam0_ud = cv2.undistortPoints(np.expand_dims(points2d_cam0, 0), camera_intrinsic0, camera_dist0,
                                               P=camera_intrinsic0)
        points2d_cam0_ud = np.squeeze(points2d_cam0_ud)

        # normalize and undistort points: cam1
        points2d_cam1_norm = cv2.undistortPoints(np.expand_dims(points2d_cam1, 0), camera_intrinsic1, camera_dist1)
        points2d_cam1_norm = np.squeeze(points2d_cam1_norm)
        points2d_cam1_ud = cv2.undistortPoints(np.expand_dims(points2d_cam1, 0), camera_intrinsic1, camera_dist1,
                                               P=camera_intrinsic1)
        points2d_cam1_ud = np.squeeze(points2d_cam1_ud)

        # fake focal length and pp
        focal = 1.0
        pp = (0.0, 0.0)

        if use_ransac:
            # RANSAC evaluates a threshold in image space to figure out the inliers during optimization;
            # when we normalize the image points forehand we need to make the threshold accordingly tighter
            threshold = 4.0 / (camera_intrinsic0[0, 0] + camera_intrinsic0[1, 1] +
                               camera_intrinsic1[0, 0] + camera_intrinsic1[1, 1])
            threshold /= 10.0  # Method is quite sensitive for the threshold chosen

            # Has its problems with "perfect" data because it doesn't estimate a final solution
            # that takes into account all samples from the optimal set of inliers
            E, mask = cv2.findEssentialMat(points2d_cam0_norm, points2d_cam1_norm, focal=focal, pp=pp, threshold=threshold,
                                           method=cv2.RANSAC)  # fails sometimes even for perfect data!!

        else:
            # LMEDs yields better results on perfect data and also does not need the threshold parameter;
            # Not sure if it has failure cases
            E, mask = cv2.findEssentialMat(points2d_cam0_norm, points2d_cam1_norm, focal=focal, pp=pp,
                                       method=cv2.LMEDS)  # Seems to work more robustly

        # Calculate relative camera pose (up to a translational scale)
        _, R, t_norm, _ = cv2.recoverPose(E, points2d_cam0_norm, points2d_cam1_norm, focal=focal, pp=pp)

        t = self.translation_scale*t_norm
        return R, t
