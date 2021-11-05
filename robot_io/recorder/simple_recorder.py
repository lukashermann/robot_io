import os
import time
import logging
import json
import datetime
from glob import glob
from pathlib import Path

from PIL import Image
import numpy as np
from robot_io.utils.utils import depth_img_to_uint16
from robot_io.utils.utils import depth_img_from_uint16


from flow_control.rgbd_camera import RGBDCamera

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


class SimpleRecorder:
    def __init__(self, env, save_dir="", n_digits=6, save_images=False):
        """
        Arguments:
            save_dir: directory in which to save
            n_digits: zero padding for files
            save_images: save .jpg image files as well
        """
        self.env = env
        self.save_dir = save_dir
        self.save_images = save_images
        self.queue = []
        self.save_frame_cnt = count_previous_frames()
        self.current_episode_filenames = []
        self.n_digits = n_digits
        os.makedirs(self.save_dir, exist_ok=True)

    def step(self, action, obs, rew, done, info):
        filename = f"frame_{self.save_frame_cnt:0{self.n_digits}d}.npz"
        filename = os.path.join(self.save_dir, filename)
        self.current_episode_filenames.append(filename)
        self.save_frame_cnt += 1
        self.queue.append((filename, action, obs, rew, done, info))

    def process_queue(self):
        """
        Process function for queue.
        Returns:
            None
        """
        for msg in self.queue:
            filename, action, obs, rew, done, info = msg
            # change datatype of depth images to save storage space
            obs = process_obs(obs)
            np.savez_compressed(filename, **obs, action=action, done=done,
                                rew=rew, info=info)

            if self.save_images:
                img = obs["rgb_gripper"]
                img_fn = filename.replace(".npz",".jpg")
                Image.fromarray(img).save(img_fn)

    def save_info(self):
        # save info
        info_fn = os.path.join(self.save_dir, "env_info.json")
        env_info = self.env.get_info()
        env_info["time"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        #env_info["T_tcp_cam"] = self.env.cam.get_extrinsic_calibration()

        with open(info_fn, 'w') as f_obj:
            json.dump(env_info, f_obj)

        self.env.camera_manager.save_calibration(self.save_dir)

    def save(self):
        length = len(self.queue)
        self.process_queue()
        self.save_info()
        print(f"saved {self.save_dir} w/ length {length}")


class PlaybackCamera(RGBDCamera):
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

    def get_image(self):
        raise NotImplementedError

    def get_intrinsics(self):
        return self.gripper_intrinsics

    #def get_projection_matrix(self):
    #    raise NotImplementedError

    #def get_camera_matrix(self):
    #    raise NotImplementedError

    #def get_dist_coeffs(self):
    #    raise NotImplementedError


class RecEnv:
    def __init__(self, file, camera_info):
        self.file = file
        # make a copy to avoid having unclosed file buffers
        with np.load(file, allow_pickle=True) as data:
            self.data = dict(data)

        gripper_attrs = dict(width=self._robot_gripper_width)
        self._gripper = type("FakeGripper", (), gripper_attrs)

        robot_attrs = dict(get_tcp_pos=self._robot_get_tcp_pos,
                           get_tcp_orn=self._robot_get_tcp_orn,
                           gripper=self._gripper)
        self.robot = type("FakeRobot", (), robot_attrs)
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

    def _robot_gripper_width(self):
        return self.data["robot_state"].item()["gripper_opening_width"]

    def _cam_get_image(self):
        return self.data["rgb_gripper"], depth_img_from_uint16(self.data["depth_gripper"])


def np_to_dict(in_dict):
    return dict([(k,v.item(0)) for k,v in in_dict.items()])

def load_rec_list(recording_dir):
    # load camera info, done so that we jus thave to load once
    camera_info = np.load(os.path.join(recording_dir, "camera_info.npz"),
                          allow_pickle=True)
    camera_info = np_to_dict(camera_info)

    files = sorted(glob(f"{recording_dir}/frame_*.npz"))
    return [RecEnv(fn, camera_info=camera_info) for fn in files]

