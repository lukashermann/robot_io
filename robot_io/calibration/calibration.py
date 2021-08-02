import numpy as np
from numpy import dot, eye, zeros, outer
from numpy.linalg import inv
import itertools
from scipy.optimize import least_squares
from robot_io.utils.utils import pos_orn_to_matrix


def pprint(arr):
    return np.array2string(arr.round(5),separator=', ')


def log(R):
    # Rotation matrix logarithm
    theta = np.arccos((R[0,0] + R[1,1] + R[2,2] - 1.0)/2.0)
    return np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * theta / (2*np.sin(theta))


def invsqrt(mat):
    u,s,v = np.linalg.svd(mat)
    return u.dot(np.diag(1.0/np.sqrt(s))).dot(v)


def calibrate(A, B):
    #transform pairs A_i, B_i
    N = len(A)
    M = np.zeros((3,3))
    for i in range(N):
        Ra, Rb = A[i][0:3, 0:3], B[i][0:3, 0:3]
        M += outer(log(Rb), log(Ra))

    Rx = dot(invsqrt(dot(M.T, M)), M.T)

    C = zeros((3*N, 3))
    d = zeros((3*N, 1))
    for i in range(N):
        Ra, ta = A[i][0:3, 0:3], A[i][0:3, 3]
        Rb, tb = B[i][0:3, 0:3], B[i][0:3, 3]
        C[3*i:3*i+3, :] = eye(3) - Ra
        d[3*i:3*i+3, 0] = ta - dot(Rx, tb)

    tx = dot(inv(dot(C.T, C)), dot(C.T, d))
    X = np.eye(4)
    X[:3, :3] = Rx
    X[:3, 3] = tx.flatten()
    return X


def calibrate_gripper_cam_peak_martin(tcp_poses, marker_poses):
    ECs = []
    for T_R_TCP, T_CAM_MARKER in zip(tcp_poses, marker_poses):
        ECs.append((np.linalg.inv(T_R_TCP), T_CAM_MARKER))

    As = []  # relative EEs
    Bs = []  # relative cams
    for pair in itertools.combinations(ECs, 2):
        (e_1, c_1), (e_2, c_2) = pair
        A = e_2 @ np.linalg.inv(e_1)
        B = c_2 @ np.linalg.inv(c_1)
        As.append(A)
        Bs.append(B)

        # symmetrize
        A = e_1 @ np.linalg.inv(e_2)
        B = c_1 @ np.linalg.inv(c_2)
        As.append(A)
        Bs.append(B)

    X = calibrate(As, Bs)
    return X


def compute_residuals_gripper_cam(x, T_robot_tcp, T_cam_marker):
    m_R = np.array([*x[6:], 1])
    T_cam_tcp = pos_orn_to_matrix(x[3:6], x[:3])

    residuals = []
    for i in range(len(T_cam_marker)):
        m_C_observed = T_cam_marker[i, :3, 3]
        m_C = T_cam_tcp @ np.linalg.inv(T_robot_tcp[i]) @ m_R
        residuals += list(m_C_observed - m_C[:3])
    return residuals


def calibrate_gripper_cam_least_squares(T_robot_tcp, T_cam_marker):

    initial_guess = np.array([0, 0, 0, 0, 0, 0, 0, 0, -0.1])
    result = least_squares(fun=compute_residuals_gripper_cam, x0=initial_guess, method='lm',
                           args=(T_robot_tcp, T_cam_marker))
    return result
