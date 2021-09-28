import random
import gym
import numpy as np
import cv2
from numpy.lib.type_check import imag

class EnvironmentNoFrameSkip:
    def __init__(self, config, frame_skip=4):
        self.frame_skip = frame_skip
        self.dims = (config.screen_height, config.screen_width)
        self.min_reward = config.min_reward
        self.max_reward = config.max_reward
        self.random_start = config.random_start
        self.env = gym.make(config.env_name)
        self.screen_buffer = [
            np.empty(shape=self.dims, dtype=np.uint8),
            np.empty(shape=self.dims, dtype=np.uint8)
        ]

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def reset(self):
        screen = self.env.reset()
        self.screen_buffer[0] = self.preprocess(screen)
        self.screen_buffer[1].fill(0)
        return self.get_observation()

    def new_random_game(self):
        screen = self.reset()
        for _ in range(random.choice(np.arange(self.random_start))):
            screen, reward, terminal, _ = self.step(0)
        return screen

    def step(self, action):
        # TODO:
        assert action >= 0 and action <= 5, "Action is out of range."
        accumulated_reward = 0
        is_done = False
        for time_step in range(self.frame_skip):
            screen, reward, terminal, info = self.env.step(action)
            accumulated_reward += reward
            if terminal == True:
                is_done = True
                break
            elif time_step >= (self.frame_skip - 2):
                index = time_step - (self.frame_skip - 2)
                self.screen_buffer[index] = self.preprocess(screen)

        accumulated_reward = max(self.min_reward, min(self.max_reward, accumulated_reward))
        observation = self.get_observation()
        return observation, accumulated_reward, is_done, info

    def preprocess(self, image):
        """ Return type: np.uint8 """
        image = image[34:194, :, :]
        image = image[::2, ::2]
        # image = np.mean(image, axis=2, keepdims=False)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.astype(np.uint8)
        # assert image.dtype == np.uint8, "Type is expected to be np.uint8"
        # image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), self.dims, interpolation=cv2.INTER_AREA)
        return image

    def get_observation(self):
        return np.max(np.stack(self.screen_buffer), axis=0)

class EnvironmentFrameSkip:
    def __init__(self, config):
        self.dims = (config.screen_height, config.screen_width)
        self.min_reward = config.min_reward
        self.max_reward = config.max_reward
        self.random_start = config.random_start
        self.env = gym.make(config.env_name)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def reset(self):
        screen = self.env.reset()
        return self.preprocess(screen)

    def new_random_game(self):
        screen = self.env.reset()
        for _ in range(random.choice(np.arange(self.random_start))):
            screen, reward, terminal, info = self.env.step(0)
        return self.preprocess(screen)

    def step(self, action):
        assert action >= 0 and action <= 5, "Action is out of range."
        screen, reward, terminal, _ = self.env.step(action)
        screen = self.preprocess(screen)
        reward = max(self.min_reward, min(self.max_reward, reward))
        return screen, reward, terminal, _

    def preprocess(self, image):
        """ Return type: np.uint8 """
        image = image[34:194, :, :]
        image = image[::2, ::2]
        # image = np.mean(image, axis=2, keepdims=False)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.astype(np.uint8)
        # assert image.dtype == np.uint8, "Type is expected to be np.uint8"
        # image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), self.dims, interpolation=cv2.INTER_AREA)
        return image