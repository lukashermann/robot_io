import os
import logging
import json
import datetime
from glob import glob
from pathlib import Path

from PIL import Image
import numpy as np
import multiprocessing as mp
from robot_io.recorder.base_recorder import BaseRecorder
from robot_io.utils.utils import depth_img_to_uint16
from robot_io.utils.utils import depth_img_from_uint16
from robot_io.utils.utils import pos_orn_to_matrix
from robot_io.cams.camera import Camera as RobotIOCamera

# A logger for this file
log = logging.getLogger(__name__)


def process_obs(obs):
    for key, value in obs.items():
        if "depth" in key:
            obs[key] = depth_img_to_uint16(obs[key])
    return obs


def unprocess_obs(obs):
    for key, value in obs.items():
        if "depth" in key:
            obs[key] = depth_img_from_uint16(obs[key])
    return obs


def count_previous_frames():
    return len(list(Path.cwd().glob("frame*.npz")))


class SimpleRecorder(BaseRecorder):
    def __init__(self, env, save_dir=None, n_digits=6, save_images=False):
        """
        SimpleRecorder is a recorder to save frames with a simple step function.
        Recordings can be loaded with load_rec_list/PlaybackEnv.

        Arguments:
            save_dir: Directory in which to save, None means working directory
            n_digits: zero padding for files
            save_images: save .jpg image files as well
        """
        self.env = env
        self.save_images = save_images
        self.queue = mp.Queue()
        self.process = mp.Process(target=self._process_queue, name="RecorderSavingProcess")
        self.process.start()
        self.running = True
        self.save_frame_cnt = count_previous_frames()
        self.current_episode_filenames = []
        self.n_digits = n_digits
        if save_dir is None:
            save_dir = os.getcwd()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self._save_info()
        logging.info("Started SimpleRecorder, saving to %s", self.save_dir)

    def step(self, obs=None, action=None, next_obs=None, rew=None, done=None, info=None, record_info=None):
        """
        Save action, next_obs, rew, and info.

        Args:
            obs: Environment observation s.
            action: Action a chosen according to obs.
            next_obs: Environment observation s' after applying action a.
            rew: Environment reward.
            done: True if environment done.
            info: Environment info.
            record_info (dict): Info by input device (e.g. when to start and stop recording).
        """
        filename = f"frame_{self.save_frame_cnt:0{self.n_digits}d}.npz"
        filename = os.path.join(self.save_dir, filename)
        self.current_episode_filenames.append(filename)
        self.save_frame_cnt += 1
        self.queue.put((filename, action, obs, rew, done, info))

    def _process_queue(self):
        """
        Process function for queue.
        """
        while True:
            msg = self.queue.get()
            if msg == "QUIT":
                self.running = False
                break
            filename, action, obs, rew, done, info = msg
            # change datatype of depth images to save storage space
            obs = process_obs(obs)
            np.savez_compressed(filename, **obs, action=action, done=done,
                                rew=rew, info=info)

            if self.save_images:
                img = obs["rgb_gripper"]
                img_fn = filename.replace(".npz", ".jpg")
                Image.fromarray(img).save(img_fn)

    def _save_info(self):
        info_fn = os.path.join(self.save_dir, "env_info.json")
        env_metadata = self.env.metadata
        env_metadata["time"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        with open(info_fn, 'w') as f_obj:
            json.dump(env_metadata, f_obj)

        self.env.camera_manager.save_calibration(self.save_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        """
            with ... as ... : logic
        Returns:
            None
        """
        if self.running:
            self.queue.put("QUIT")
            self.process.join()


class PlaybackCamera(RobotIOCamera):
    def __init__(self, camera_info, get_image_fn):
        """
        Provide a camera API from a recording.
        """
        self.camera_info = camera_info
        self.gripper_extrinsic_calibration = camera_info["gripper_extrinsic_calibration"]
        self.gripper_intrinsics = camera_info["gripper_intrinsics"]

        self.calibration = self.gripper_intrinsics
        # set get_image function
        self.get_image = get_image_fn

        self.resolution = (640, 480)  # warning: must be tuple.
        self.crop_coords = None
        self.resize_resolution = None

    def get_image(self):
        raise NotImplementedError

    def get_intrinsics(self):
        return self.gripper_intrinsics

    def get_extrinsic_calibration(self):
        return self.gripper_extrinsic_calibration

    #def get_projection_matrix(self):
    #    raise NotImplementedError

    #def get_camera_matrix(self):
    #    raise NotImplementedError

    #def get_dist_coeffs(self):
    #    raise NotImplementedError


def load_camera_info(recording_dir):
    """
    Load camera info, in function so that we can just load once.
    """
    camera_info = np.load(os.path.join(recording_dir, "camera_info.npz"),
                          allow_pickle=True)
    #npz file object -> dict
    camera_info = dict(list(camera_info.items()))
    # TODO(max): zero-length npz arrays -> dict
    camera_info["gripper_intrinsics"] = camera_info["gripper_intrinsics"].item()
    return camera_info


class PlaybackEnv:
    """
    PlaybackEnv loads a recorded environment state. It then tries to provide
    the same interface for accessing state information as a "live" env.
    This extends to the robot and the camera.

    e.g.
    env.robot.get_tcp_pose()

    See `load_rec_list` for loading several frames at once.
    """
    def __init__(self, dir_or_file, camera_info="load"):
        if os.path.isfile(dir_or_file):
            file = dir_or_file
        else:
            raise NotImplemented
        self.file = file

        # make a copy to avoid having unclosed file buffers
        with np.load(file, allow_pickle=True) as data:
            self.data = dict(data)

        # robot & grippers
        gripper_attrs = dict(width=self._robot_gripper_width)
        self._gripper = type("FakeGripper", (), gripper_attrs)
        robot_attrs = dict(get_tcp_pos=self._robot_get_tcp_pos,
                           get_tcp_orn=self._robot_get_tcp_orn,
                           get_tcp_pos_orn=self._robot_get_tcp_pos_orn,
                           get_tcp_pose=self._robot_get_tcp_pose,
                           gripper=self._gripper)
        self.robot = type("FakeRobot", (), robot_attrs)

        if camera_info == "load":
            camera_info = load_camera_info(os.path.dirname(file))
        self.cam = PlaybackCamera(camera_info, self._cam_get_image)

    def get_action(self):
        action = self.data["action"].item()
        if action["motion"] == [0, 0, 0]:
            print("Deprecation warning [0,0,0] action.")
            action["motion"] = (np.zeros(3), np.array([1, 0, 0, 0]), 1)
        if isinstance(action["motion"][0], tuple):
            action["motion"] = (np.array(action["motion"][0]),
                                np.array(action["motion"][1]),action["motion"][2])
        return action

    def get_robot_state(self):
        return self.data["robot_state"].item()

    def _robot_get_tcp_pos(self):
        return self.data["robot_state"].item()["tcp_pos"]

    def _robot_get_tcp_orn(self):
        return self.data["robot_state"].item()["tcp_orn"]

    def _robot_get_tcp_pos_orn(self):
        tmp = self.data["robot_state"].item()
        return tmp["tcp_pos"], tmp["tcp_orn"]

    def _robot_get_tcp_pose(self):
        return pos_orn_to_matrix(*self._robot_get_tcp_pos_orn())

    def _robot_gripper_width(self):
        return self.data["robot_state"].item()["gripper_opening_width"]

    def _cam_get_image(self):
        return self.data["rgb_gripper"], depth_img_from_uint16(self.data["depth_gripper"])


def load_rec_list(recording_dir):
    """
    Loads several pre-recorded demonstration frames.

    Returns:
        list of PlaybackEnvs
    """
    # load camera info, done so that we jus thave to load once
    camera_info = load_camera_info(recording_dir)
    files = sorted(glob(f"{recording_dir}/frame_*.npz"))
    return [PlaybackEnv(fn, camera_info=camera_info) for fn in files]
