import os
import time
from pathlib import Path

import numpy as np
import multiprocessing as mp
import threading
import logging
from pathlib import Path
from robot_io.utils.utils import TextToSpeech, depth_img_to_uint16
# A logger for this file
log = logging.getLogger(__name__)


def process_obs(obs):
    for key, value in obs:
        if "depth" in key:
            obs[key] = depth_img_to_uint16(obs[key])
    return obs


def count_previous_frames():
    return len(list(Path.cwd().glob("frame*.npz")))


class VrRecorder:
    def __init__(self, n_digits):
        self.recording = False
        self.queue = mp.Queue()
        self.process = mp.Process(target=self.process_queue, name="MultiprocessingStorageWorker")
        self.process.start()
        self.running = True
        self.save_frame_cnt = count_previous_frames()
        self.tts = TextToSpeech()
        self.current_episode_filenames = []
        self.n_digits = n_digits
        self.delete_thread = None
    
    def step(self, action, obs, record_info):
        if record_info["trigger_release"] and not self.recording and not self.is_deleting:
            self.recording = True
            self.tts.say("start recording")
            self.current_episode_filenames = []
        elif record_info["trigger_release"] and self.recording:
            self.recording = False
            self.save(action, obs, True)
            self.tts.say("finish recording")
        if record_info["hold_event"]:
            if self.recording:
                self.recording = False
            self.delete_last_episode()

        if self.recording:
            self.save(action, obs, False)

    @property
    def is_deleting(self):
        return self.delete_thread is not None and self.delete_thread.is_alive()

    def delete_last_episode(self):
        self.delete_thread = threading.Thread(target=self._delete_last_episode, daemon=True)
        self.delete_thread.start()

    def _delete_last_episode(self):
        log.info("Delete episode")
        while not self.queue.empty():
            log.info("Wait until files are saved")
            time.sleep(0.01)
        num_frames = len(self.current_episode_filenames)
        self.tts.say(f"Deleting last episode with {num_frames} frames")
        for filename in self.current_episode_filenames:
            os.remove(filename)
        self.tts.say("Finished deleting")
        self.save_frame_cnt -= num_frames
        self.current_episode_filenames = []

    def save(self, action, obs, done):
        filename = f"frame_{self.save_frame_cnt:0{self.n_digits}d}.npz"
        self.current_episode_filenames.append(filename)
        self.save_frame_cnt += 1
        self.queue.put((filename, action, obs, done))

    def process_queue(self):
        """
        Process function for queue.
        Returns:
            None
        """
        while True:
            msg = self.queue.get()
            if msg == "QUIT":
                self.running = False
                break
            filename, action, obs, done = msg
            # change datatype of depth images to save storage space
            obs = process_obs(obs)
            np.savez(filename, **obs, action=action, done=done)

    def __enter__(self):
        """
            with ... as ... : logic
        Returns:
            None
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
            with ... as ... : logic
        Returns:
            None
        """
        if self.running:
            self.queue.put("QUIT")
            self.process.join()
