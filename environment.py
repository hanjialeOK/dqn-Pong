import gym
import numpy as np

def preprocess(image):
    image = image[34:194, :, :]
    image = image[::2, ::2]
    image = np.mean(image, axis=2, keepdims=False)
    image = image/256
    return image

class Environment:
    def __init__(self, config):
        self.min_reward = config.min_reward
        self.max_reward = config.max_reward
        self.env = gym.make('Pong-v0')

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def reset(self):
        screen = self.env.reset()
        return preprocess(screen)

    def step(self, action):
        assert action == 0 or action == 1, "action is out of range."
        screen, reward, terminal, _ = self.env.step(action + 2)
        screen = preprocess(screen)
        reward = max(self.min_reward, min(self.max_reward, reward))
        return screen, reward, terminal, _