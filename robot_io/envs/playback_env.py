"""
The PlaybackEnv class provides a way to load a recording and then to provide the same
interface for accessing state information as from a "live" env. This extends to the
robot and the camera.

Example:
    tcp_pos, tcp_orn = env.robot.get_tcp_pos_orn()  # normal interface for accessing env
    pb_env = PlaybackEnv.freeze(env)                # save env state
    pb_pos, pb_orn = pb_env.robot.get_tcp_pos_orn() # access recording like live env.
"""
import logging
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from robot_io.cams.camera import Camera as RobotIOCamera
from robot_io.recorder.simple_recorder import unprocess_obs, SimpleRecorder
from robot_io.utils.utils import pos_orn_to_matrix


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

        resolution_from_intr = self.gripper_intrinsics["width"], self.gripper_intrinsics["height"]
        self.resolution = resolution_from_intr
        self.crop_coords = None
        self.resize_resolution = None

    def get_image(self):
        raise NotImplementedError

    def get_intrinsics(self):
        return self.gripper_intrinsics

    def get_extrinsic_calibration(self):
        return self.gripper_extrinsic_calibration

    # def get_projection_matrix(self):
    #    raise NotImplementedError

    # def get_camera_matrix(self):
    #    raise NotImplementedError

    # def get_dist_coeffs(self):
    #    raise NotImplementedError

    def is_similar(self, other_pb_cam):
        same_intrinsics = self.get_intrinsics() == other_pb_cam.get_intrinsics()
        same_extrinsics = self.get_extrinsic_calibration() == other_pb_cam.get_extrinsic_calibration()
        if same_intrinsics and same_extrinsics:
            return True
        return False


def load_camera_info(recording_dir):
    """
    Load camera info, in function so that we can just load once.
    """
    camera_info = np.load(Path(recording_dir) / "camera_info.npz", allow_pickle=True)
    # npz file object -> dict
    camera_info = dict(list(camera_info.items()))
    # TODO(max): zero-length npz arrays -> dict
    camera_info["gripper_intrinsics"] = camera_info["gripper_intrinsics"].item()
    return camera_info


class PlaybackEnvStep:
    """
    PlaybackEnvStep loads a recorded environment state. It then tries to provide
    the same interface for accessing state information as a "live" env.
    This extends to the robot and the camera.

    e.g.
    env.robot.get_tcp_pose()

    See `PlaybackEnv` for loading several frames at once.
    """

    def __init__(self, file, camera_info="load"):
        """
        Arguments:
            file: filename of single frame, e.g. frame_000000.npz
        """
        # make a copy to avoid having unclosed file buffers
        with np.load(file, allow_pickle=True) as data:
            self.data = unprocess_obs(dict(data))

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
            camera_info = load_camera_info(Path(file).parent)
        self.cam = PlaybackCamera(camera_info, self._cam_get_image)

    def get_action(self, component=None):
        action = self.data["action"].item()

        if action is None:
            return None

        # TODO(max): this is fucking ridiculous, default to tuples.
        if isinstance(action["motion"], tuple):
            action["motion"] = list(action["motion"])

        for i in range(len(action["motion"])):
            if isinstance(action["motion"][i], np.ndarray):
                action["motion"][i] = tuple(action["motion"][i])

            if isinstance(action["motion"][i], (list, tuple)):
                if len(action["motion"][i]) == 1:
                    action["motion"][i] = action["motion"][i][0]

        if isinstance(action["motion"], np.ndarray):
            action["motion"] = action["motion"].tolist()
        if action["motion"] == [0, 0, 0]:
            print("Deprecation warning [0,0,0] action.")
            action["motion"] = (np.zeros(3), np.array([1, 0, 0, 0]), 1)
        if isinstance(action["motion"][0], tuple):
            action["motion"] = (np.array(action["motion"][0]),
                                np.array(action["motion"][1]), action["motion"][2])

        if component is None:
            return action
        elif component == "gripper":
            if action is None:
                return None
            else:
                return action["motion"][2]
        else:
            raise ValueError

    # def get_obs(self):
    #    set_trace()
    #    #action, done, rew, info

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
        return self.data["rgb_gripper"], self.data["depth_gripper"]


class PlaybackEnv:
    def __init__(self, recording_dir, load="all", n_digits=6):
        """
        A masked recording where we step between frames that are included in keep_dict,
        skipping over those omitted.

        Arguments:
            recording_dir: directory where the recording is found
            load: "all" or list of ints which are converted to filenames
        """
        # load camera info, done so that we jus have to load once
        if not Path(recording_dir).is_dir():
            raise FileNotFoundError(f"directory not found: {recording_dir}")

        cam_info = load_camera_info(recording_dir)
        cnt2fn = lambda cnt: Path(recording_dir) / f"frame_{cnt:0{n_digits}d}.npz"
        if load == "all":
            # load all frames, first have to find which ones there are.
            files = sorted(glob(f"{recording_dir}/frame_*.npz"))
            self.steps = [PlaybackEnvStep(fn, camera_info=cam_info) for fn in files]
            self.keep_indexes = sorted(self.keep_dict.keys())

        elif isinstance(load, list):
            # load keyframes, first have to find which ones they are.
            self.keep_indexes = sorted(load)
            self.steps = {}
            for cnt in self.keep_indexes:
                self.steps[cnt] = PlaybackEnvStep(cnt2fn(cnt), camera_info=cam_info)
        else:
            raise ValueError

        self.index_keep = 0  # where we are in self.keep_dict
        self.index = self.keep_indexes[0]  # where to index in self.steps

    def reset(self):
        self.index_keep = 0
        self.index = self.keep_indexes[0]

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, index):
        return self.steps[index]

    def __getattr__(self, name):
        index = self.__getattribute__("index")
        return getattr(self.steps[index], name)

    def step(self):
        self.index_keep += 1
        self.index = self.keep_indexes[np.clip(self.index_keep, 0, len(self.keep_indexes) - 1)]
        if isinstance(self.steps, list):
            assert not self.index > len(self.steps) - 1

    def to_list(self):
        return self.steps

    def get_max_frame(self):
        if isinstance(self.steps, list):
            return len(self.steps) - 1
        if isinstance(self.steps, dict):
            return sorted(self.steps.keys())[-1]

    # In theory this function should be in the PlaybackRecorder class, in practice it's a
    # bit more convenient to have it here.
    @staticmethod
    def freeze(env, reward=0, done=False):
        """
        Create a static view of a single env step w/ extra info for servoing.
        """
        obs, info = env._get_obs()

        with TemporaryDirectory() as tmp_dir_name:
            simp_rec = SimpleRecorder(env, tmp_dir_name)
            action = dict(motion=(None, None, 1))
            simp_rec.step(action, obs, reward=reward, done=done, info=info)
            simp_rec.save()
            demo_pb = PlaybackEnv(tmp_dir_name)

        return demo_pb

