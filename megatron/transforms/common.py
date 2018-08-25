import numpy as np
from ..core import Transformation
from ..utils import initializer


class TimeSeries(Transformation):
    @initializer
    def __init__(self, window_size, time_axis=1, reverse=False):
        super().__init__()

    def transform(self, X):
        steps = [np.roll(X, i, axis=0) for i in range(self.window_size)]
        out = np.moveaxis(np.stack(steps), 0, self.time_axis)[self.window_size:]
        return np.flip(out, axis=-1) if self.reverse else out
