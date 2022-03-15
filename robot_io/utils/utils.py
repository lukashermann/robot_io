import math
from enum import Enum
from pathlib import Path

import git
import sys
import time
import quaternion
import hydra
import numpy as np
import multiprocessing as mp
from scipy.spatial.transform.rotation import Rotation as R


def z_angle_between(a, b):
    """
    :param a: 3d vector
    :param b: 3d vector
    :return: signed angle between vectors around z axis (right handed rule)
    """
    return math.atan2(b[1], b[0]) - math.atan2(a[1], a[0])


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(v1_u @ v2_u, -1.0, 1.0))


def angle_between_angles(a, b):
    diff = b - a
    return (diff + np.pi) % (2 * np.pi) - np.pi


def scipy_quat_to_np_quat(quat):
    """xyzw to wxyz"""
    return np.quaternion(quat[3], quat[0], quat[1], quat[2])


def np_quat_to_scipy_quat(quat):
    """wxyz to xyzw"""
    return np.array([quat.x, quat.y, quat.z, quat.w])


def euler_to_quat(euler_angles):
    """xyz euler angles to xyzw quat"""
    return R.from_euler('xyz', euler_angles).as_quat()


def quat_to_euler(quat):
    """xyz euler angles to xyzw quat"""
    return R.from_quat(quat).as_euler('xyz')


def xyz_to_zyx(orn):
    """xyz euler angles to zyx euler"""
    return orn[::-1]


def pos_orn_to_matrix(pos, orn):
    """
    :param pos: np.array of shape (3,)
    :param orn: np.array of shape (4,) -> quaternion xyzw
                np.quaternion -> quaternion wxyz
                np.array of shape (3,) -> euler angles xyz
    :return: 4x4 homogeneous transformation
    """
    mat = np.eye(4)
    if isinstance(orn, np.quaternion):
        orn = np_quat_to_scipy_quat(orn)
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 4:
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 3:
        mat[:3, :3] = R.from_euler('xyz', orn).as_matrix()
    mat[:3, 3] = pos
    return mat


def orn_to_matrix(orn):
    mat = np.eye(3)
    if isinstance(orn, np.quaternion):
        orn = np_quat_to_scipy_quat(orn)
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 4:
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 3:
        mat[:3, :3] = R.from_euler('xyz', orn).as_matrix()
    return mat


def matrix_to_orn(mat):
    """
    :param mat: 4x4 homogeneous transformation
    :return: tuple(position: np.array of shape (3,), orientation: np.array of shape (4,) -> quaternion xyzw)
    """
    return R.from_matrix(mat[:3, :3]).as_quat()


def matrix_to_pos_orn(mat):
    """
    :param mat: 4x4 homogeneous transformation
    :return: tuple(position: np.array of shape (3,), orientation: np.array of shape (4,) -> quaternion xyzw)
    """
    orn = R.from_matrix(mat[:3, :3]).as_quat()
    pos = mat[:3, 3]
    return pos, orn


def to_world_frame(rel_action_pos, rel_action_orn, tcp_orn):
    T_world_tcp_old = orn_to_matrix(tcp_orn)
    pos_w_rel = T_world_tcp_old[:3, :3] @ rel_action_pos
    T_tcp_new_tcp_old = orn_to_matrix(rel_action_orn)

    T_world_tcp_new = T_world_tcp_old @ np.linalg.inv(T_tcp_new_tcp_old)
    orn_w_new = matrix_to_orn(T_world_tcp_new)
    orn_w_new = quat_to_euler(orn_w_new)
    orn_w_rel = tcp_orn - orn_w_new
    orn_w_rel = np.where(orn_w_rel < -np.pi, orn_w_rel + 2 * np.pi, orn_w_rel)
    orn_w_rel = np.where(orn_w_rel > np.pi, orn_w_rel - 2 * np.pi, orn_w_rel)
    return pos_w_rel, orn_w_rel


def to_tcp_frame(rel_action_pos, rel_action_orn, tcp_orn):
    T_tcp_world = np.linalg.inv(orn_to_matrix(tcp_orn))
    pos_tcp_rel = T_tcp_world @ rel_action_pos
    new_orn_w = tcp_orn + rel_action_orn
    T_world_tcp = orn_to_matrix(tcp_orn)
    T_world_tcp_new = orn_to_matrix(new_orn_w)
    T_tcp_new_tcp_old = np.linalg.inv(T_world_tcp_new) @ T_world_tcp
    orn_tcp_rel = quat_to_euler(matrix_to_orn(T_tcp_new_tcp_old))
    orn_tcp_rel = np.where(orn_tcp_rel < -np.pi, orn_tcp_rel + 2 * np.pi, orn_tcp_rel)
    orn_tcp_rel = np.where(orn_tcp_rel > np.pi, orn_tcp_rel - 2 * np.pi, orn_tcp_rel)
    return pos_tcp_rel, orn_tcp_rel

import logging

log = logging.getLogger(__name__)


class TextToSpeech:
    def __init__(self):
        self.queue = mp.Queue()
        self.process = mp.Process(target=self.tts_worker, name="TTS_worker")
        self.process.daemon = True
        self.process.start()

    def say(self, text):
        log.info(text)
        self.queue.put(text)

    def tts_worker(self):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        while True:
            text = self.queue.get()
            engine.say(text)
            engine.runAndWait()


class FpsController:
    def __init__(self, freq):
        self.loop_time = 1.0 / freq
        self.prev_time = time.time()

    def step(self):
        current_time = time.time()
        delta_t = current_time - self.prev_time
        if delta_t < self.loop_time:
            time.sleep(self.loop_time - delta_t)
        self.prev_time = time.time()


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


def get_git_root(path):
    git_repo = git.Repo(Path(path), search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return Path(git_root)


def depth_img_to_uint16(depth_img, max_depth=4):
    depth_img = np.clip(depth_img, 0, max_depth)
    return (depth_img / max_depth * (2 ** 16 - 1)).astype('uint16')


def depth_img_from_uint16(depth_img, max_depth=4):
    return depth_img.astype('float') / (2 ** 16 - 1) * max_depth


class ReferenceType(Enum):
    ABSOLUTE = 0
    RELATIVE = 1
    JOINT = 2