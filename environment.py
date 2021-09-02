import random
import gym
import numpy as np

def preprocess(image):
    image = image[34:194, :, :]
    image = image[::2, ::2]
    image = np.mean(image, axis=2, keepdims=False)
    image = image/255
    return image

class Environment:
    def __init__(self, config):
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
        return preprocess(screen)

    def new_random_game(self):
        screen = self.env.reset()
        for _ in range(random.choice(np.arange(self.random_start))):
            screen, reward, terminal, _ = self.env.step(0 + 1)
        return preprocess(screen)

    def step(self, action):
        assert action >= 0 and action <= 5, "action is out of range."
        screen, reward, terminal, _ = self.env.step(action)
        screen = preprocess(screen)
        reward = max(self.min_reward, min(self.max_reward, reward))
        return screen, reward, terminal, _