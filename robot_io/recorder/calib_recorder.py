import os
import time
from pathlib import Path

import numpy as np
import multiprocessing as mp
import threading
import logging
from robot_io.utils.utils import TextToSpeech

# A logger for this file
log = logging.getLogger(__name__)


class CalibRecorder:
    def __init__(self, n_digits):
        self.queue = mp.Queue()
        self.process = mp.Process(target=self.process_queue, name="MultiprocessingStorageWorker")
        self.process.start()
        self.running = True
        self.save_frame_cnt = 0
        self.tts = TextToSpeech()
        self.prev_done = False
        self.current_episode_filenames = []
        self.n_digits = n_digits

    def step(self, tcp_pose, marker_pose, record_info):
        if record_info["trigger_release"]:
            self.tts.say("pose sampled")
            self.save(tcp_pose, marker_pose)

    def save(self, tcp_pose, marker_pose):
        filename = f"frame_{self.save_frame_cnt:0{self.n_digits}d}.npz"
        self.current_episode_filenames.append(filename)
        self.save_frame_cnt += 1
        self.queue.put((filename, tcp_pose, marker_pose))

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
            filename, tcp_pose, marker_pose = msg
            print("saving ", filename, os.getcwd())
            np.savez(filename, tcp_pose=tcp_pose, marker_pose=marker_pose)

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
