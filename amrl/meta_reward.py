from collections import deque
import numpy as np

class MetaReward:
    def __init__(self, window=1000):
        self.buf = deque(maxlen=window)
        self.hist = []

    def update(self, r):
        self.buf.append(r)

    def compute(self):
        avg = np.mean(self.buf) if self.buf else 0.0
        self.hist.append(avg)
        if len(self.hist) > 1:
            return self.hist[-1] - self.hist[-2]
        return 0.0
