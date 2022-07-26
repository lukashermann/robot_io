import os
import time

import numpy as np
import pygame

from robot_io.robot_interface.base_robot_interface import GripperInterface as GI
from robot_io.utils.utils import FpsController


class KeyboardInput:
    """
    Keyboard input class.
    For 5-DOF control, use WASD to move the robot in the xy-plane, X/Z to move the robot in z-direction and
    Q/E to rotate around the z-axis.
    For 7-DOF control, additionally use the arrow keys to pitch and roll the end-effector.

    Args:
        act_type: Currently only "continuous" is implemented.
        initial_gripper_state: "open" or "closed".
        dv: Position offset for relative actions in meter.
        drot: Orientation offset for relative actions in rad.
        mode: "5dof" or "7dof"
    """

    def __init__(self, act_type="continuous", initial_gripper_state='open', dv=0.01, drot=0.2, reference_frame='tcp', mode='5dof', **kwargs):
        assert act_type == 'continuous'
        assert mode in ('5dof', '7dof')
        assert reference_frame in ('tcp', 'world')
        self.reference_frame = reference_frame
        self.pressed_keys = []
        self.mode = mode
        self.done = False

        self.speeds = [1.0, 0.5, .25, .1, .05]
        self.speed_index = 0

        print("Starting Keyboard Input:")
        print("dv =", dv)
        print("drot =", drot)

        self.movement_keys = {ord('w'): (dv, 0, 0, 0, 0, 0, 0),
                              ord('s'): (-dv, 0, 0, 0, 0, 0, 0),
                              ord('a'): (0, -dv, 0, 0, 0, 0, 0),
                              ord('d'): (0, dv, 0, 0, 0, 0, 0),
                              ord('x'): (0, 0, dv, 0, 0, 0, 0),
                              ord('z'): (0, 0, -dv, 0, 0, 0, 0),
                              ord('i'): (0, 0, -dv, 0, 0, 0, 0),
                              ord('k'): (0, 0, dv, 0, 0, 0, 0),
                              ord('e'): (0, 0, 0, 0, 0, drot, 0),
                              ord('q'): (0, 0, 0, 0, 0, -drot, 0),
                              }
        if mode == '7dof':
            self.movement_keys[1073741906] = (0, 0, 0, 0, -drot, 0, 0)
            self.movement_keys[1073741905] = (0, 0, 0, 0, drot, 0, 0)
            self.movement_keys[1073741904] = (0, 0, 0, -drot, 0, 0, 0)
            self.movement_keys[1073741903] = (0, 0, 0, drot, 0, 0, 0)

        self.gripper_key = ord('\t')
        self.prev_gripper_key_pressed = False
        # convert str or GripperState to GripperState
        self.gripper_state = GI.to_gripper_state(initial_gripper_state)

        self.modify_keys = {ord('h'): self.print_help,
                            1073741910: self.decrease_motion,
                            1073741911: self.increase_motion}

        # init pygame stuff
        icon_fn = os.path.join(os.path.dirname(__file__), "assets/keyboard_icon.png")
        image = pygame.image.load(icon_fn)

        screen = pygame.display.set_mode(image.get_size())
        screen.fill((255, 255, 255))
        screen.blit(image, (0, 0))
        pygame.display.update()

        self.reset()

    def reset(self):
        self.done = False
        self.pressed_keys = []

    def get_action(self):
        """
        Get the keyboard action dictionary.

        Returns:
            action (dict): Keyboard action.
            record_info: None (to be consistent with other input devices).
        """
        raw_action, extra_keys = self._handle_keyboard_events()
        action = {"motion": np.split(raw_action, [3, 6]),
                  "ref": "rel",
                  "path": "lin",
                  "blocking": False,
                  "impedance": False}
        # To be compatible with vr input actions. For now there is nothing to pass as record info
        record_info = {"done": self.done}
        if len(extra_keys) > 0:
            record_info["extra_keys"] = extra_keys

        return action, record_info

    def _handle_keyboard_events(self):
        """
        Detect keyboard events and compose action.

        Returns:
            Action as a numpy array of shape (7,).
        """
        pressed_once_keys = []
        extra_keys = set()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in self.movement_keys:
                    self.pressed_keys.append(event.key)
                elif event.key == 27:  # escape
                    self.done = True
                elif event.key == self.gripper_key and not self.prev_gripper_key_pressed:
                    self.prev_gripper_key_pressed = True
                    self.gripper_state = GI.toggle(self.gripper_state)
                elif event.key in self.modify_keys:
                    self.modify_keys[event.key]()
                else:
                    print("Unassigned key:", event.key)
                    extra_keys.add(event.unicode)


            elif event.type == pygame.KEYUP:
                if event.key in self.movement_keys:
                    self.pressed_keys.remove(event.key)
                elif event.key == self.gripper_key:
                    self.prev_gripper_key_pressed = False

        actions = [self.movement_keys[key] for key in self.pressed_keys]
        actions += [(0, 0, 0, 0, 0, 0, 0)]
        action = np.sum(actions, axis=0, dtype=float)
        # scale movements, but not gripper action.
        action *= self.speeds[self.speed_index]

        if self.reference_frame == "world":
            action[1] *= -1
            action[2] *= -1
            action[4] *= -1
            action[5] *= -1

        # gripper action
        action[-1] = self.gripper_state.value

        pygame.display.update()

        assert action.shape == (7,)
        return action, extra_keys

    @staticmethod
    def print_all_events():
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                print(event)

    @staticmethod
    def print_help():
        print("Keyboard Input:")
        print("use WASD to move the robot in the xy-plane, X/Z to move the robot in z-direction and")
        print("Q/E to rotate around the z-axis.")
        print("Use the arrow keys to pitch and roll the end-effector.")

    def decrease_motion(self):
        if self.speed_index < len(self.speeds)-1:
            self.speed_index += 1
            print("decreasing motion: now scaling speeds by", self.speeds[self.speed_index])
        else:
            print("minimum speed already reached")

    def increase_motion(self):
        if self.speed_index > 0:
            self.speed_index -= 1
            print("increasing motion: now scaling speeds by", self.speeds[self.speed_index])
        else:
            print("maximum speed already reached")


def keyboard_control(env):
    """
    Use this to free the robot from a stuck position.

    Args:
        env: RobotEnv
    """
    kb = KeyboardInput(dv=0.005, drot=0.01, reference_frame='tcp', mode='7dof')
    kb.print_help()
    print("Press ESC to leave keyboard control.")
    done = False
    fps = FpsController(20)
    while not done:
        action, info = kb.get_action()
        print(action)
        done = info["done"]
        env.step(action)
        fps.step()


if __name__ == "__main__":
    keyboard = KeyboardInput()
    while not keyboard.done:
        # keyboard.print_all_events()
        print(keyboard.get_action())
        time.sleep(.5)
