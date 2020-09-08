import os
import numpy as np
import pygame

class KeyboardInput:
    """
    Keyboard input classs.

    enables wasd type control of robot
    """

    def __init__(self, act_type='continuous'):
        assert act_type == 'continuous'

        self.pressed_keys = []
        self.relevant_keys = {ord('w'): (0, -1, 0, 0, 0),
                              ord('s'): (0, 1, 0, 0, 0),
                              ord('a'): (1, 0, 0, 0, 0),
                              ord('d'): (-1, 0, 0, 0, 0),
                              273:      (0, 0, 1, 0, 0),
                              274:      (0, 0, -1, 0, 0),
                              275:      (0, 0, 0, 1, 0),
                              276:      (0, 0, 0, -1, 0),
                              ord('\t'): (0, 0, 0, 0, -1)
                              }
        # init pygame stuff
        screen = pygame.display.set_mode((256, 256))
        icon_fn = os.path.join(os.path.dirname(__file__),"assets/keyboard_icon.png")
        image = pygame.image.load(icon_fn)
        screen.fill((255, 255, 255))
        screen.blit(image, (0, 0))
        pygame.display.update()

    def handle_keyboard_events(self):
        """
        handles keyboard actions
        Returns:
            action: currently only 5dof
        """
        pressed_once_keys = []
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in self.relevant_keys:
                    pressed_once_keys.append(event.key)
                    self.pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in self.relevant_keys:
                    self.pressed_keys.remove(event.key)

        # pressed once
        actions_once = [self.relevant_keys[key] for key in pressed_once_keys]
        actions_once += [(0, 0, 0, 0, 0)]
        # pressed long
        actions = [self.relevant_keys[key] for key in self.pressed_keys]
        actions += [(0, 0, 0, 0, 0)]
        action = 0.5*np.sum(actions_once, axis=0) + np.sum(actions, axis=0)
        assert action.shape == (5,)
        return action

    def print_all_events(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                print(event)
               
if __name__ == "__main__":
    keyboard = KeyboardInput()
    while True:
        keyboard.print_all_events()
