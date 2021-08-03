import numpy as np

class History:
    def __init__(self, config):
        self.history = np.zeros(shape=(config.history_length, config.screen_height, config.screen_width), dtype=np.float32)

    def add(self, screen):
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    def get(self):
        return np.expand_dims(self.history, axis=0)