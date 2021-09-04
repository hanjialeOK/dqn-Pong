import random
import gym
import numpy as np
import cv2

class Environment:
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
            screen, reward, terminal, _ = self.env.step(2)
        return self.preprocess(screen)

    def step(self, action):
        assert action >= 0 and action <= 5, "action is out of range."
        screen, reward, terminal, _ = self.env.step(action)
        screen = self.preprocess(screen)
        reward = max(self.min_reward, min(self.max_reward, reward))
        return screen, reward, terminal, _

    def preprocess(self, image):
        # image = image[::2, ::2]
        # image = np.mean(image, axis=2, keepdims=False)
        image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        image = image/255
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, self.dims, interpolation=cv2.INTER_AREA)
        return image