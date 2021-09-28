import numpy as np

class History:
    def __init__(self, config):
        self.dims = (config.screen_height, config.screen_width)
        self.history_length = config.history_length
        self.history = np.zeros((self.history_length, ) + self.dims, dtype=np.uint8)

    def add(self, screen):
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    def get(self):
        return np.expand_dims(self.history, axis=0)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen_width", type=int, default=40, help="Screen width after resize.")
    parser.add_argument("--screen_height", type=int, default=52, help="Screen height after resize.")
    parser.add_argument("--history_length", type=int, default=4, help="How many screen frames form a state.")
    parser.add_argument("--loops", type=int, default=1000000, help="Number of loops in testing.")
    args = parser.parse_args()

    import numpy as np
    mem = History(args)
    for i in range(args.loops):
        mem.add(np.zeros((args.screen_height, args.screen_width)))
        if i >= args.history_length:
            state = mem.get()