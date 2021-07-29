import gym
import random
import numpy as np
import matplotlib.pyplot as plt

def preprocess(image):
    env1 = gym.make('LunarLander-v2')
    env1.reset()
    image = env1.render(mode='rgb_array')
    image = image[0:336, 132:468]
    image = image[::4, ::4]
    image = np.mean(image, axis=2, keepdims=False)
    image = image/256
    plt.imshow(image)
    plt.savefig('test.png')
    # return image
