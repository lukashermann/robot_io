import math
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


def matrix_to_pos_orn(mat):
    """
    :param mat: 4x4 homogeneous transformation
    :return: tuple(position: np.array of shape (3,), orientation: np.array of shape (4,) -> quaternion xyzw)
    """
    orn = R.from_matrix(mat[:3, :3]).as_quat()
    pos = mat[:3, 3]
    return pos, orn

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
