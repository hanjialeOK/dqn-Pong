import numpy as np
import random

class ReplayMemory:
    # TODO: -> None
    def __init__(self, config) -> None:
        self.memory_size = config.memory_size
        self.batch_size = config.batch_size
        self.history_length = config.history_length
        self.actions = np.empty((self.memory_size, 1), dtype=np.uint8)
        self.screens = np.empty((self.memory_size, config.screen_height, config.screen_width), dtype=np.float16)
        self.rewards = np.empty((self.memory_size, 1), dtype=np.int8)
        self.terminals = np.empty((self.memory_size, 1), dtype=np.bool_)
        self.dims = (config.screen_height, config.screen_width)
        self.current = 0
        self.count = 0

        # TODO: pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)
        self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)

    def add(self, action, screen, reward, terminal):
        assert screen.shape == self.dims, "screen's shape is unexpected."
        self.actions[self.current, ...] = action
        self.screens[self.current, ...] = screen
        self.rewards[self.current, ...] = reward
        self.terminals[self.current, ...] = terminal
        # TODO:
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def getState(self, index):
        assert self.count > 0, " memory is empty."
        assert index >= self.history_length - 1 and index < self.count, "index is out of range."
        return self.screens[(index - (self.history_length - 1)):(index + 1), ...]

    def sample(self):
        assert self.count > self.history_length, "memory is almost empty."
        indexes = []
        while len(indexes) < self.batch_size:
            while True:
                # sample one index
                index = random.randint(self.history_length, self.count - 1)
                # if wraps over current pointer
                if index >= self.current and index - self.current < self.history_length:
                    continue
                # if wraps over eposide
                if self.terminals[(index - self.history_length):index].any() == True:
                    continue
                break
            self.prestates[len(indexes), ...] = self.getState(index - 1)
            self.poststates[len(indexes), ...] = self.getState(index)
            # TODO:
            indexes.append(index)

        actions = self.actions[indexes, ...]
        rewards = self.rewards[indexes, ...]
        terminals = self.terminals[indexes, ...]

        return self.prestates, actions, rewards, self.poststates, terminals
