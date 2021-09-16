import random
import gym
import numpy as np
import cv2

class Environment:
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
        assert action >= 0 and action <= 5, "action is out of range."
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
        image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        # image = image/255
        image = cv2.resize(image, self.dims, interpolation=cv2.INTER_AREA)
        return image

    def get_observation(self):
        return np.max(np.stack(self.screen_buffer), axis=0) / 255