import os
import copy
import json
import logging
import datetime
import subprocess
from pathlib import Path

from PIL import Image
import numpy as np

from robot_io.utils.utils import depth_img_to_uint16
from robot_io.utils.utils import depth_img_from_uint16

# A logger for this file
log = logging.getLogger(__name__)


def process_obs(obs):
    for key, value in obs.items():
        if "depth" in key:
            assert obs[key].dtype == np.float32
            obs[key] = depth_img_to_uint16(obs[key])
    return obs


def unprocess_obs(obs):
    for key, value in obs.items():
        if "depth" in key:
            obs[key] = depth_img_from_uint16(obs[key]).astype(np.float32)
    return obs


def unprocess_seg(pixel):
    """
    This is code specific to pybullet segmentation masks.

    Pybullet segmenetations mix object uids and link ids. Untangle these.
    https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/segmask_linkindex.py
    """
    obj_uid = pixel & ((1 << 24) - 1)
    link_index = (pixel >> 24) - 1
    return obj_uid, link_index


class SimpleRecorder:
    def __init__(self, env, save_dir="", n_digits=6):
        """
        PlaybackRecorder is a recorder to save frames with a simple step function.
        Recordings can be loaded with PlaybackEnv/PlaybackEnvStep.

        Arguments:
            save_dir: directory in which to save
            n_digits: zero padding for files
        """
        self.env = env
        self.save_dir = save_dir
        self.queue = []
        self.save_frame_cnt = len(list(Path.cwd().glob("frame*.npz")))
        self.current_episode_filenames = []
        self.n_digits = n_digits
        os.makedirs(self.save_dir, exist_ok=True)

    def step(self, action, obs, reward, done, info):
        filename = f"frame_{self.save_frame_cnt:0{self.n_digits}d}.npz"
        filename = Path(self.save_dir) / filename
        self.current_episode_filenames.append(filename)
        self.save_frame_cnt += 1
        self.queue.append((filename, action, obs, reward, done, info))

    def process_queue(self, image_path=None):
        """
        Process function for queue.
        """
        for msg in self.queue:
            filename, action, obs, rew, done, info = msg
            # change datatype of depth images to save storage space
            obs = process_obs(copy.deepcopy(obs))
            np.savez_compressed(filename, **obs, action=action, done=done,
                                rew=rew, info=info)

            if image_path is not None:
                img = obs["rgb_gripper"]
                image_fn = Path(image_path) / Path(filename).name
                image_fn = image_fn.replace(".npz", ".jpg")
                Image.fromarray(img).save(image_fn)

    def save_info(self):
        metadata_fn = Path(self.save_dir) / "env_metadata.json"
        env_metadata = dict(self.env.metadata)
        env_metadata["time"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        with open(metadata_fn, 'w') as f_obj:
            json.dump(env_metadata, f_obj)

        # will create camera_info.npz file
        self.env.camera_manager.save_calibration(self.save_dir)

    def save(self, save_images=None, save_video=False):
        """
        Save the recording

        Arguments:
            save_images: None, False, True (None set to True if save_video)
            save_video: False or True, requires ffmepg
        """
        # can only save videos if we save images
        if save_images is None and save_video is True:
            save_images = True
        if save_images:
            image_path = self.save_dir
        else:
            image_path = None

        length = len(self.queue)
        self.process_queue(image_path=image_path)

        if save_video:
            self.save_video()

        self.save_info()

        print(f"saved {self.save_dir} w/ length {length}")

    def save_video(self):
        subproc_cmd = f'ffmpeg -framerate 8 -i {self.save_dir}/frame_%06d.jpg -r 25 -pix_fmt yuv420p ' \
                      f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {self.save_dir}/video.mp4'

        # Run subprocess using the command
        subprocess.run(subproc_cmd, check=True, shell=True)

