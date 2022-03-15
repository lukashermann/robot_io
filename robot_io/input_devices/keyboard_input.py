import os
from enum import Enum

import numpy as np
import pygame


class GripperState(Enum):
    OPEN = 1
    CLOSED = -1


class KeyboardInput:
    """
    Keyboard input classs.

    enables wasd type control of robot
    """

    def __init__(self, act_type="continuous", initial_gripper_state='open', dv=0.01, drot=0.2, mode='5dof', **kwargs):
        assert act_type == 'continuous'
        assert initial_gripper_state in ('open', 'closed')
        assert mode in ('5dof', '7dof')
        self.pressed_keys = []
        self.mode = mode
        self.movement_keys = {ord('w'): (dv, 0, 0, 0, 0, 0, 0),
                              ord('s'): (-dv, 0, 0, 0, 0, 0, 0),
                              ord('a'): (0, -dv, 0, 0, 0, 0, 0),
                              ord('d'): (0, dv, 0, 0, 0, 0, 0),
                              ord('x'): (0, 0, dv, 0, 0, 0, 0),
                              ord('z'): (0, 0, -dv, 0, 0, 0, 0),
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
        if initial_gripper_state == 'open':
            self.gripper_state = GripperState.OPEN
        else:
            self.gripper_state = GripperState.CLOSED

        # init pygame stuff
        screen = pygame.display.set_mode((256, 256))
        icon_fn = os.path.join(os.path.dirname(__file__), "assets/keyboard_icon.png")
        image = pygame.image.load(icon_fn)
        screen.fill((255, 255, 255))
        screen.blit(image, (0, 0))
        pygame.display.update()

    def get_action(self):

        raw_action = self.handle_keyboard_events()
        action = {"motion": np.split(raw_action, [3, 6]), "ref": "rel"}
        # To be compatible with vr input actions. For now there is nothing to pass as record info
        record_info = None

        return action, record_info

    def handle_keyboard_events(self):
        """
        handles keyboard actions
        Returns:
            action: currently only 5dof
        """
        pressed_once_keys = []
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in self.movement_keys:
                    self.pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
                elif event.key == self.gripper_key and not self.prev_gripper_key_pressed:
                    self.prev_gripper_key_pressed = True
                    if self.gripper_state == GripperState.OPEN:
                        self.gripper_state = GripperState.CLOSED
                    elif self.gripper_state == GripperState.CLOSED:
                        self.gripper_state = GripperState.OPEN

            elif event.type == pygame.KEYUP:
                if event.key in self.movement_keys:
                    self.pressed_keys.remove(event.key)
                elif event.key == self.gripper_key:
                    self.prev_gripper_key_pressed = False

        actions = [self.movement_keys[key] for key in self.pressed_keys]
        actions += [(0, 0, 0, 0, 0, 0, 0)]
        action = np.sum(actions, axis=0)

        # gripper action
        action[-1] = self.gripper_state.value

        assert action.shape == (7,)
        return action

    def print_all_events(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                print(event)


if __name__ == "__main__":
    keyboard = KeyboardInput()
    while True:
        # keyboard.print_all_events()
        print(keyboard.get_action())
