import time
import hydra
import numpy as np

import gym

from robot_io.utils.utils import timeit


class RobotEnv(gym.Env):
    def __init__(self,
                 robot,
                 camera_manager_cfg,
                 workspace_limits):
        """
        :param robot:
        :param workspace_limits: workspace bounding box [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        """
        self.robot = robot
        self.workspace_limits = workspace_limits

        self.camera_manager = hydra.utils.instantiate(camera_manager_cfg)

    def reset(self, target_pos=None, target_orn=None):
        """
        Reset robot to neutral position.
        """
        if target_pos is not None and target_orn is not None:
            self.robot.move_cart_pos_abs_ptp(target_pos, target_orn)
        else:
            self.robot.move_to_neutral()
        return self._get_obs()

    def _get_obs(self):
        """
        :return: dictionary with image obs and state obs
        """
        obs = self.camera_manager.get_images()
        obs['robot_state'] = self.robot.get_state()
        return obs

    def get_reward(self, obs, action):
        return 0

    def get_termination(self, obs):
        return False

    def get_info(self, obs, action):
        info = {}
        return info

    def step(self, action):
        """
        Execute one action on the robot.
        :param action: {"motion": (position, orientation, gripper_action), "ref": "rel"/"abs"}
                       a dict with the key 'motion' which is a cartesian motion tuple
        :              and the key 'ref' which specifies if the motion is absolute or relative
        :return: obs, reward, done, info
        """
        if action is None:
            return self._get_obs(), 0, False, {}
        assert isinstance(action, dict) and len(action['motion']) == 3

        target_pos, target_orn, gripper_action = action['motion']
        ref = action['ref']

        if ref == "abs":
            target_pos = self._restrict_workspace(target_pos)
            # TODO: use LIN for panda
            self.robot.move_async_cart_pos_abs_lin(target_pos, target_orn)
        elif ref == "rel":
            self.robot.move_async_cart_pos_rel_ptp(target_pos, target_orn)
        else:
            raise ValueError

        if gripper_action == 1:
            self.robot.open_gripper()
        elif gripper_action == -1:
            self.robot.close_gripper()
        else:
            raise ValueError

        obs = self._get_obs()

        reward = self.get_reward(obs, action)

        termination = self.get_termination(obs)

        info = self.get_info(obs, action)

        return obs, reward, termination, info

    def _restrict_workspace(self, target_pos):
        """
        :param target_pos: cartesian target position
        :return: clip target_pos at workspace limits
        """
        return np.clip(target_pos, self.workspace_limits[0], self.workspace_limits[1])

    def render(self, mode='human'):
        if mode == 'human':
            self.camera_manager.render()
